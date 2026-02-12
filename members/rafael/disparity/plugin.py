from interface import SatellitePlugin
import numpy as np
import os
import shutil
from pathlib import Path

from . import constants as C
from .processing import (
    generate_rectified,
)
from .disparity import disparity_map
from .preprocessing import get_crop_area_from_kml, generate_cropped
from .utils import normalise_for_display, open_tiff_file

from .pair_selector import PairSelector, PairCandidate, ImageCandidate

#TODO: Compare results with what AMES is able to do
# TODO: do a parameter sweep for bothh wls sigma and lambda, but also for MODE_SGBM, MODE_HH, MODE_SGBM_3WAY if you had the time
PREFIX = "[Multi-day 3D Point Cloud]"


class HeightMapExtractor(SatellitePlugin):
    """Height map and 3d Point cloud extraction from WV3 multi-day satellite images.

    This plugin auto-discovers multi-day stereo pairs from data/WV3/PAN directory.
    It does not use a user-selected image - only a KML file for the region of interest.
    """

    # Flag to indicate this plugin doesn't need image selection
    requires_image = False

    @property
    def name(self):
        return "Multi-day 3D Point Cloud"

    def run(self, kml_path: Path, 
            is_debug_mode: bool = C.IS_DEBUG_MODE,
            is_debug_pair: bool = C.IS_DEBUG_PAIR,
            is_one_random_pair: bool = C.IS_ONE_RANDOM_PAIR,
            n: int = C.N):
        layers = []

        kml_path_str = str(kml_path)

        if os.path.exists(C.TEMP_PATH):
            shutil.rmtree(C.TEMP_PATH)
        os.makedirs(C.TEMP_PATH, exist_ok=False)
        
        with open(C.TEMP_PATH + "/log.txt", 'w') as f:
            f.writelines("3D Point Cloud started")
            try:
                f.writelines(f"loading images from: {C.WV3_PATH}\n")
                pair_selector = PairSelector(C.WV3_PATH)
                pair_selector.discover_images()
                pairs: list[PairCandidate] = pair_selector.select_pairs()
            
                os.makedirs(C.TMP_STEREO_OUTPUT_PATH, exist_ok=False)
                os.makedirs(C.TMP_CROPPED_IMAGES_PATH, exist_ok=False)
                os.makedirs(C.TMP_DISPARITY_DEBUG_PATH, exist_ok=False)
                
                # PREPROCESSING
                f.write(f"preprocessing pairs")
                if is_debug_pair:
                    pairs = [pair for pair in pairs if (pair.img1.filename == C.PAIR_DECENT_RESULTS[0][0] or pair.img1.filename == C.PAIR_DECENT_RESULTS[0][1])
                             and (pair.img2.filename == C.PAIR_DECENT_RESULTS[0][1] or pair.img2.filename == C.PAIR_DECENT_RESULTS[0][0])]
                    print("DEBUG:" + str(len(pairs)))
                elif is_one_random_pair:
                    pairs = [pairs[np.random.randint(0, min(n, len(pairs)))]]
                else:
                    pairs = pairs[:n]

                for pair in pairs:
                    for img in [pair.img1, pair.img2]:
                        output_path =C.TMP_CROPPED_IMAGES_PATH
                        output_name = img.filename + '.tif'
                        
                        if not os.path.exists(img.path):
                            f.write(f"image not found  {img.path}")
                            return [(np.zeros((100, 100)), {"name": "error: image not found"}, "image")]
                    
                        if os.path.exists(os.path.join(output_path, output_name)):
                            continue
                        try:
                            crop_area = get_crop_area_from_kml(img, kml_path_str)
                            f.write(f"crop area for {img.filename}: {crop_area}")
                            generate_cropped(img, output_path, output_name, crop_area)
                            f.write(f"generated cropped image for {img.filename} at {output_path}")

                        except Exception as e:
                            f.write(f"error: Cropping failed for {img.filename}: {str(e)}")
                            return [(np.zeros((100, 100)), {"name": f"error: {str(e)}"}, "image")]
                
                # Rectify the pair using AMP Stereo
                # the orignal paper argues for an affine camera model
                # which i also tried, but due to the rpc transformation 
                # required, taking into account the azimuth angle, indicence angle,
                # sun elevation angle, etc. the results were bad
                # i still believe they used a complex package that itself implements 
                # an affine camera model as it is mentioned in the original paper
                # and is written by the same people who wrote the original paper
                # https://github.com/centreborelli/s2p
                # s2p could not be run due to it never being updated to work with Arm OSX
                # dependencies were failing
                # ASP is using a perspective camera model and is mantained by NASA

                for pair_id, pair in enumerate(pairs):
                    f.write("ASP stereo rectification...")
                    out_path = generate_rectified(pair, pair_id, C.TMP_STEREO_OUTPUT_PATH)
                    f.write(f"Rectification complete, output: {out_path}")
                
                # DISPARITY MAP
                    f.write("Generating disparity map...")
                    disparity, validity_mask, photoconsistency = disparity_map(
                        pair, pair_id, 
                        C.TMP_STEREO_OUTPUT_PATH, C.TMP_CROPPED_IMAGES_PATH, C.TMP_DISPARITY_DEBUG_PATH
                    )
                    
                    f.write("Disparity map generated successfully")
                    
                    if is_debug_mode:
                    # DISPLAY
                        f.write("Loading  basic cropped image for display...")
                        cropped_left = open_tiff_file(os.path.join(C.TMP_CROPPED_IMAGES_PATH, pair.img1.cropped_name))
                        valid_mask = cropped_left > 0
                        cropped_display = normalise_for_display(cropped_left, valid_mask)
                        layers.append((cropped_display, {"name": f"{PREFIX} Input Left", "colormap": "gray"}, "image"))

                        cropped_right = open_tiff_file(os.path.join(C.TMP_CROPPED_IMAGES_PATH, pair.img2.cropped_name))
                        valid_mask = cropped_right > 0
                        cropped_display = normalise_for_display(cropped_right, valid_mask)
                        layers.append((cropped_display, {"name": f"{PREFIX} Input Right", "colormap": "gray"}, "image"))

                        rectified_left_path = os.path.join(C.TMP_STEREO_OUTPUT_PATH, str(pair_id), 'results', 'out-L.tif')
                        if os.path.exists(rectified_left_path):
                            rectified_left = open_tiff_file(rectified_left_path)
                            valid_mask = rectified_left > 0
                            rectified_display = normalise_for_display(rectified_left, valid_mask)
                            layers.append((rectified_display, {"name": f"{PREFIX} Rectified Left", "colormap": "gray"}, "image"))

                        rectified_right_path = os.path.join(C.TMP_STEREO_OUTPUT_PATH, str(pair_id), 'results', 'out-R.tif')
                        if os.path.exists(rectified_right_path):
                            rectified_right = open_tiff_file(rectified_right_path)
                            valid_mask = rectified_right > 0
                            rectified_display = normalise_for_display(rectified_right, valid_mask)
                            layers.append((rectified_display, {"name": f"{PREFIX} Rectified Right", "colormap": "gray"}, "image"))
                        
                    # openCv stores disparity as integer/16 bit fixed point
                    height_map = -disparity.astype(float) / 16.0 
                    
                    # Sentinels are large values, valid range is within MAX_DISP/2
                    limit = C.MAX_DISP / 2
                    valid_mask = np.isfinite(height_map) & (np.abs(height_map) <= limit) & validity_mask
                    
                    # couldn't make it work by passing arguments to the napari renderer, that should also be possible
                    # use a relative height map instead
                    # fixes the "moving gradient" issue in visualization
                    y_indices, x_indices = np.where(valid_mask)
                    z_values = height_map[valid_mask]

                    P = np.stack([x_indices, y_indices, z_values], axis=1)
                    center = np.mean(P, axis=0)
                    P_centered = P - center
                        
                    U, S, Vh = np.linalg.svd(P_centered, full_matrices=False)
                    normal = Vh[2] 

                    if np.dot(normal, np.array([0,0,1])) < 0:
                        normal = -normal
                        
                    # project points to this relative height
                    height_rel = np.dot(P_centered, normal)
                        
                    # update the disparity map image for display
                    z_values = height_rel
                    height_map[valid_mask] = z_values

                    height_map_normalized = normalise_for_display(height_map, valid_mask)
                    height_map_normalized[~valid_mask] = np.nan
                    layers.append((height_map_normalized, {"name": f"{PREFIX} Disparity", "colormap": "turbo", "scale": (1, 1)}, "image"))

                    h_min = np.percentile(z_values, 2)
                    h_max = np.percentile(z_values, 98)
                    div = h_max - h_min + 1e-6
                    h_norm = (z_values - h_min) / div
                    h_norm = np.clip(h_norm, 0, 1)
                    property_table = {
                        'height': h_norm
                    }

                    # set the ground level to 0 (move 2nd percentile to 0)
                    z_values = z_values - h_min
                    points_coords = np.stack([z_values, y_indices, x_indices], axis=1)

                    # shows the confidence based on the cost of the disparity map match
                    valid_mask = photoconsistency > 0
                    photoconsistency_normalized = normalise_for_display(photoconsistency, valid_mask)
                    layers.append((photoconsistency_normalized, {"name": f"{PREFIX} Photoconsistency", "colormap": "turbo", "scale": (1, 1)}, "image"))

                    black_canvas = (~valid_mask).astype(np.float32)

                    custom_mask_colormap = {
                        "colors": [
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]
                        ],
                        "name": "mask_blackout",
                        "interpolation": "linear"
                    }
                    layers.append((
                        black_canvas,
                        {
                            "name": f"{PREFIX} Invalid Mask",
                            "colormap": custom_mask_colormap,
                            "scale": (1, 1),
                            "contrast_limits": [0, 1]
                        },
                        "image"
                    ))

                    # Output: 3D Point Cloud
                    layers.append((
                        points_coords,
                        {
                            "name": f"{PREFIX} 3D Point Cloud",
                            "size": 2,
                            "properties": property_table,
                            "scale": (1, 1, 1),
                            "opacity": 0.8,
                            "face_colormap": "turbo",
                            "face_color": "height",
                        },
                        "points"
                    ))
                f.writelines(f"Added {len(layers)} layers to Napari")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                f.writelines(error_msg)
                return [(np.ones((100, 100)), {"name": f"Error: {str(e)}"}, "image")]
            
            return layers
