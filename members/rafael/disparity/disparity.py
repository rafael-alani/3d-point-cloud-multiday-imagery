import cv2
import numpy as np

from .constants import BLOCK_SIZE_DISP
from .utils import save_disparity, save_equalized_disparity, open_tiff_file, save_tiff_file

from .processing import (
    normalise_image, post_process_undefined, warp_coordinates, photoconsistency_map, add_margin, remove_margin
)
from .constants import (
    MAX_DISP,
    WLS_LAMBDA, WLS_SIGMA, IS_DEBUG_MODE
)
import os
from .pair_selector import PairCandidate

from PIL import Image

# based on https://docs.opencv.org/3.4/d3/d14/tutorial_ximgproc_disparity_filtering.html
# and the docs of openCV
def disparity_map(pair: PairCandidate, pair_id, tmp_stereo_output_path,
                  tmp_cropped_images_path, tmp_disparity_debug_path):
                                       
    path = os.path.join(tmp_stereo_output_path, str(pair_id), 'results/')
    tmp_disparity_debug_path = os.path.join(tmp_disparity_debug_path, str(pair_id))
    max_disp = MAX_DISP
    left = open_tiff_file(pair.rectified_pair_path + '-L.tif')
    right = open_tiff_file(pair.rectified_pair_path + '-R.tif')
    
    left_infos_path = os.path.join(tmp_cropped_images_path, pair.img1.cropped_name + '.npy')
    if os.path.exists(left_infos_path):
        left_infos_np = np.load(left_infos_path)
    else:
        raise FileNotFoundError(f"Could not find left npy file: {left_infos_path}")
    

    # uses an affine matrix to align the images, we need it later
    exr_path = path + 'out-align-L.txt'
    if os.path.exists(exr_path):
        print("Existant affine matrix will be used!")
        left_M = np.loadtxt(exr_path)
    else:
        print(f"exr path: {exr_path}")
        left_M = np.eye(3)
    print(left_M)
    
    if IS_DEBUG_MODE:
        os.makedirs(tmp_disparity_debug_path, exist_ok=True)
        max_pair = max(left.max(), right.max())
        
        raw_left = left.copy()
        raw_right = right.copy()
        
        raw_left[raw_left < 0] = 0
        raw_right[raw_right < 0] = 0

        if max_pair > 0:
            raw_left = (raw_left * 255.0 / max_pair).astype(np.uint8)
            raw_right = (raw_right * 255.0 / max_pair).astype(np.uint8)
        
        Image.fromarray(raw_left).save(os.path.join(tmp_disparity_debug_path, f'1-L-RAW.png'))
        Image.fromarray(raw_right).save(os.path.join(tmp_disparity_debug_path, f'1-R-RAW.png'))
    

    # SANDARDIZE BRIGHTNESS
    #  different days, different times of the day and times of the year
    # values can be very off from one anthoer
    bound_left = normalise_image(left)
    bound_right = normalise_image(right)
    
    left = bound_left.copy()
    right = bound_right.copy()
    
    left_undefined = left < 0
    right_undefined = right < 0
    
    left_invalid = post_process_undefined(left_undefined.copy(), max_disp)
    right_invalid = post_process_undefined(right_undefined.copy(), max_disp)

    left[left < 0] = 0
    right[right < 0] = 0
    bound_left[bound_left < 0] = 0
    bound_right[bound_right < 0] = 0

    # opencv needs uint8
    left *= 255
    right *= 255
    bound_left *= 255
    bound_right *= 255
    left = left.astype('uint8')
    right = right.astype('uint8')
    bound_left = bound_left.astype('uint8')
    bound_right = bound_right.astype('uint8')
    
    left = filter_im(left)
    right = filter_im(right)
    
    margin = max_disp
    left = add_margin(left, margin)
    right = add_margin(right, margin)
    bound_left = add_margin(bound_left, margin)
    bound_right = add_margin(bound_right, margin)

    # new masks should be be ware the the margin is also invalid/undefined
    left_undefined = add_margin(left_undefined, margin, True)
    right_undefined = add_margin(right_undefined, margin, True)
    left_invalid = add_margin(left_invalid, margin, True)
    right_invalid = add_margin(right_invalid, margin, True)
    
    if IS_DEBUG_MODE:
        Image.fromarray(left).save(os.path.join(tmp_disparity_debug_path, f'1.1-L.png'))
        Image.fromarray(right).save(os.path.join(tmp_disparity_debug_path, f'1.1-R.png'))
        Image.fromarray(bound_left).save(os.path.join(tmp_disparity_debug_path, f'1.1-L-std.png'))
        Image.fromarray(bound_right).save(os.path.join(tmp_disparity_debug_path, f'1.1-R-std.png'))
        Image.fromarray((left_invalid * 255).astype(np.uint8)).save(os.path.join(tmp_disparity_debug_path, f'1.1-L-invalid.png'))
        Image.fromarray((right_invalid * 255).astype(np.uint8)).save(os.path.join(tmp_disparity_debug_path, f'1.1-R-invalid.png'))

    # DISPARITY
    # def disparity_map(pair: PairCandidate, pair_id, tmp_stereo_output_path,
    #                   tmp_cropped_images_path, tmp_disparity_debug_path):
    left_disp, right_disp, left_matcher, right_matcher = disparity_images(left, right, max_disp)
    
    if IS_DEBUG_MODE:
        save_disparity(left_disp, os.path.join(tmp_disparity_debug_path, f'2-left_disparity.png'))
        save_disparity(right_disp, os.path.join(tmp_disparity_debug_path, f'2-right_disparity.png'))
        consistency = left_right_consistency(left_disp / 16.0, right_disp / 16.0, -(max_disp // 2)) < 1.5
        save_disparity(consistency, os.path.join(tmp_disparity_debug_path, f'2-consistency.png'))

    # FIRST PASS WLS
    left_disp_wls1, left_confidence_wls1 = weighted_least_square_filter(
        left_matcher, bound_left, bound_right,
        left_disp, right_disp, left_invalid, right_invalid, max_disp
    )
    right_disp_wls1, right_confidence_wl1 = weighted_least_square_filter(
        left_matcher, bound_right, bound_left,
        right_disp, left_disp, right_invalid, left_invalid, max_disp
    )

    valid_disp_range = (-(max_disp // 2) * 16, (max_disp // 2) * 16)
    if IS_DEBUG_MODE:
        save_disparity(left_confidence_wls1, os.path.join(tmp_disparity_debug_path, f'3-left_confidence.png'), 1)
        save_disparity(right_confidence_wl1, os.path.join(tmp_disparity_debug_path, f'3-right_confidence.png'), 1)
        save_disparity(left_disp_wls1, os.path.join(tmp_disparity_debug_path, f'3-left_filtered_disparity.png'), valid_range=valid_disp_range)
        save_disparity(right_disp_wls1, os.path.join(tmp_disparity_debug_path, f'3-right_filtered_disparity.png'), valid_range=valid_disp_range)
    
    # SECOND PASS WLS
    left_disp_wls2, left_confidence_wls2 = weighted_least_square_filter(
        left_matcher, bound_left, bound_right,
        left_disp_wls1, right_disp_wls1, left_invalid, right_invalid, max_disp
    )
    
    right_disp_wls2, right_confidence_wls2 = weighted_least_square_filter(
        left_matcher, bound_right, bound_left,
        right_disp_wls1, left_disp_wls1, right_invalid, left_invalid, max_disp
    )
    
    left_right_consistency_init = left_right_consistency(left_disp / 16.0, right_disp / 16.0, -(max_disp // 2)) < 3
    left_right_consistency_wls1 = left_right_consistency(left_disp_wls1 / 16.0, right_disp_wls1 / 16.0, -(max_disp // 2)) < 3
    left_right_consistency_wls2 = left_right_consistency(left_disp_wls2 / 16.0, right_disp_wls2 / 16.0, -(max_disp // 2)) < 3
    
    photoconsistency = photoconsistency_map(left, right, left_disp_wls2 / 16.0, -(max_disp // 2))
    
    if IS_DEBUG_MODE:
        save_disparity(left_right_consistency_init, os.path.join(tmp_disparity_debug_path, f'3.1-f-consistency-init.png'))
        save_disparity(left_right_consistency_wls1, os.path.join(tmp_disparity_debug_path, f'3.1-f-consistency-1.png'))
        save_disparity(left_right_consistency_wls2, os.path.join(tmp_disparity_debug_path, f'3.1-f-consistency-2.png'))
        save_disparity(left_confidence_wls2, os.path.join(tmp_disparity_debug_path, f'3.1-left_confidence_post.png'))
        save_disparity(left_disp_wls2, os.path.join(tmp_disparity_debug_path, f'3.1-left_filtered_disparity_post.png'), valid_range=valid_disp_range)
        Image.fromarray((photoconsistency * 255).astype(np.uint8)).save(
            os.path.join(tmp_disparity_debug_path, f'4-photoconsistency.png')
        )
    
    disparity = left_disp_wls2
    
    disparity = remove_margin(disparity, margin)
    left_undefined = remove_margin(left_undefined, margin)
    photoconsistency = remove_margin(photoconsistency, margin)
    left_right_consistency_init = remove_margin(left_right_consistency_init, margin)
    left_right_consistency_wls1 = remove_margin(left_right_consistency_wls1, margin)
    left_right_consistency_wls2 = remove_margin(left_right_consistency_wls2, margin)
    
    # Filter out WLS sentinel values from the valid mask
    # WLS sets invalid regions to -(max_disp) * 16 or (max_disp) * 16
    min_valid_disp = -(max_disp // 2) * 16
    max_valid_disp = (max_disp // 2) * 16
    disparity_in_range = np.logical_and(disparity >= min_valid_disp, disparity <= max_valid_disp)
    left_defined_np = np.logical_and(np.logical_not(left_undefined), disparity_in_range)
    
    original_width = left_infos_np[2] - left_infos_np[0]
    original_height = left_infos_np[3] - left_infos_np[1]
        
    (Yfrom, Xfrom) = np.where(left_defined_np)
    Xto, Yto, Xfrom, Yfrom = warp_coordinates(Xfrom, Yfrom, np.linalg.inv(left_M), original_width, original_height)
    
    final_defined_positions = np.logical_and(
        np.logical_and(Xto >= left_infos_np[4], Xto <= original_width - left_infos_np[5]), 
        np.logical_and(Yto >= left_infos_np[6], Yto <= original_height - left_infos_np[7])
    )

    Xdefined = Xfrom[final_defined_positions]
    Ydefined = Yfrom[final_defined_positions]
    
    final_defined = np.zeros(left_defined_np.shape, dtype=bool)
    final_defined[Ydefined, Xdefined] = True
    
    if IS_DEBUG_MODE:
        Image.fromarray((final_defined * 255).astype(np.uint8)).save(
            os.path.join(tmp_disparity_debug_path, f'5-final_defined.png')
        )
        final_disparity = disparity.copy()
        save_equalized_disparity(final_disparity, os.path.join(tmp_disparity_debug_path, f'5-left_filtered_disparity_final.png'), final_defined)

    outf = np.zeros((3, disparity.shape[0], disparity.shape[1]))
    main_channel = 0
    
    outf[main_channel,:,:] = -disparity
    outf[main_channel,:,:] /= 16.0
    outf[2,:,:] = final_defined

    np.savez_compressed(path + 'consistency', photoconsistency=photoconsistency,
                        left_right_consistency_wls1=left_right_consistency_wls1,
                        left_right_consistency_wls2=left_right_consistency_wls2,
                        left_right_consistency_init=left_right_consistency_init)
    save_tiff_file(path + '5-out-F.tif', outf)
    
    return disparity, final_defined, photoconsistency


def left_right_consistency(left_disp, right_disp, min_disp, max_disp=80):
    consistency_mat = np.zeros(left_disp.shape)
    nb_positions = left_disp.shape[0] * left_disp.shape[1]
    positions = np.arange(nb_positions)
    xs = (positions % left_disp.shape[1]).astype(int)
    ys = (positions // left_disp.shape[1]).astype(int)
    
    reshaped_left_disp_np = left_disp.reshape(nb_positions)
    disp_xs = np.round(xs - reshaped_left_disp_np).astype(int)
    undefined_values = np.logical_or(
        np.logical_or(np.logical_or(np.isnan(reshaped_left_disp_np), disp_xs < 0), 
                      disp_xs >= left_disp.shape[1]), 
        reshaped_left_disp_np < min_disp
    )
    disp_xs[undefined_values] = xs[undefined_values]
    
    diff = np.abs(right_disp[ys, disp_xs] + left_disp[ys, xs])
    diff[undefined_values] = max_disp
    
    consistency_mat[ys, xs] = diff
    
    return consistency_mat


# TODO: if results are still not satisfying
def filter_im(image):
    return image


# Main functions
# https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
# https://stackoverflow.com/questions/62627109/how-do-you-use-opencvs-disparitywlsfilter-in-python
# as we're dealing with satelite imagery one image doesn't necessarly have
# only positive disparities, due to different angles, etc both are possible?
def disparity_images(left, right, max_disp, scale = BLOCK_SIZE_DISP, uniqueness_ratio = 0):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-(max_disp // 2),
        numDisparities=max_disp,
        blockSize=scale
    )
    P1 = int(round(8 * scale * scale))
    P2 = int(round(32 * scale * scale))

    left_matcher.setMode(0)
    left_matcher.setP1(P1)
    left_matcher.setP2(P2)
    left_matcher.setUniquenessRatio(uniqueness_ratio)
    left_matcher.setSpeckleWindowSize(0)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(left, right)
    right_disp = right_matcher.compute(right, left)

    return left_disp, right_disp, left_matcher, right_matcher

# edge guided filter to smoothen disparity
# https://stackoverflow.com/questions/62627109/how-do-you-use-opencvs-disparitywlsfilter-in-python
def weighted_least_square_filter(left_matcher, left, right, ori_left_disp, ori_right_disp,
                                 left_invalid, right_invalid, max_disp, disc_radius=3, lrc_threshold=24):
    left_disp = ori_left_disp.copy()
    right_disp = ori_right_disp.copy()

    invalid_values = -np.ones(ori_left_disp.shape) * (max_disp + 10) * 16
    invalid_values[:, (invalid_values.shape[1] // 2):] = (max_disp + 20) * 16

    left_to_remove = left_invalid.copy()
    right_to_remove = right_invalid.copy()
    left_disp[left_to_remove] = invalid_values[left_to_remove]
    right_disp[right_to_remove] = invalid_values[right_to_remove]

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(WLS_LAMBDA)
    wls_filter.setSigmaColor(WLS_SIGMA)
    wls_filter.setDepthDiscontinuityRadius(disc_radius)
    wls_filter.setLRCthresh(lrc_threshold)
    disparity = np.zeros(left_disp.shape)
    disparity = wls_filter.filter(left_disp, left, disparity, right_disp,
                                  (0, 0, left_disp.shape[1], left_disp.shape[0]), right)

    confidence_np = wls_filter.getConfidenceMap()

    return disparity, confidence_np