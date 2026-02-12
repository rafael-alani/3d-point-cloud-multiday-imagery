import cv2
import numpy as np

def spectral_residual_global_detection(img: np.ndarray, tile_size=512, padding=64):

    # Handle NaN values
    img = np.nan_to_num(img, nan=0.0)

    if img.ndim == 3:
        # Handle channel-first format (C, H, W) -> (H, W, C)
        if img.shape[0] <= 4 and img.shape[1] > 4 and img.shape[2] > 4:
            img = np.transpose(img, (1, 2, 0))
        # Convert multi-channel image to grayscale by averaging all channels
        gray = np.mean(img, axis=2)
    else:
        gray = img

    # Normalize to 0-255 uint8 for cv2 operations
    gmin, gmax = gray.min(), gray.max()
    if gmax - gmin < 1e-10:
        gray = np.zeros_like(gray, dtype=np.uint8)
    else:
        gray = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)

    h, w = gray.shape
    full_saliency_map = np.zeros((h, w), dtype=np.uint8)
    
    #our image size is way higher than the papers, for the log spectrum Hou used a 64x64 downscale
    # due to this we will also split our image, into more reasonable tile sizes
    # use padding to detect objects that cross the boundaries
    # i arrived at this method after simply applying 1 salience map lead to poor results
    # we also add padding to not have disjoint saliency maps when joined back
    resize_ratio = 64.0 / tile_size 

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            
            y_pad_top = max(0, y - padding)
            y_pad_bottom = min(h, y + tile_size + padding)
            x_pad_left = max(0, x - padding)
            x_pad_right = min(w, x + tile_size + padding)
            tile = gray[y_pad_top:y_pad_bottom, x_pad_left:x_pad_right]
            th, tw = tile.shape

            small_w = int(tw * resize_ratio)
            small_h = int(th * resize_ratio)
            tile_small = cv2.resize(tile, (small_w, small_h), interpolation=cv2.INTER_AREA)

            # FFT block of equations
            f = np.fft.fft2(tile_small)
            #eq 5 and 7
            log_amplitude = np.log(np.abs(np.fft.fftshift(f)) + 1e-10)
            #eq 6
            phase = np.angle(np.fft.fftshift(f))
            
            # eq 3
            avg_log_amp = cv2.blur(log_amplitude, (3, 3))
            #eq 8
            spectral_residual = log_amplitude - avg_log_amp
            
            #eq 9
            f_ishift = np.fft.ifftshift(np.exp(spectral_residual + 1j * phase))
            saliency = np.abs(np.fft.ifft2(f_ishift))
            saliency = saliency ** 2
            saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
            saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # scale size back
            saliency = cv2.resize(saliency, (tw, th))
            
            # padding is only used to calculate border values, remove
            valid_y_start = y - y_pad_top
            valid_y_end = valid_y_start + min(tile_size, h - y)
            valid_x_start = x - x_pad_left
            valid_x_end = valid_x_start + min(tile_size, w - x)

            valid_saliency = saliency[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            place_h, place_w = valid_saliency.shape
            full_saliency_map[y:y+place_h, x:x+place_w] = valid_saliency
    
    bounding_boxes = []
    
    # eq10, E(S(x)) * 3
    mean_saliency = np.mean(full_saliency_map)
    threshold_val = min(mean_saliency * 3.0, 255)
    _, thresh = cv2.threshold(full_saliency_map, threshold_val, 255, cv2.THRESH_BINARY)
    
    # fill in gaps
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # TODO: play with values 
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append((x, y, w, h))

    return img, full_saliency_map, bounding_boxes