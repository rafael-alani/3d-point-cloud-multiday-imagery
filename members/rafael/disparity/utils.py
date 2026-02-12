import numpy as np
import cv2
from PIL import Image
from osgeo import gdal

from skimage import exposure


def normalise_for_display(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image.astype(float)
    
    p2, p98 = np.percentile(image[mask], [2, 98])
    image = np.clip((image - p2) / (p98 - p2 + 1e-6), 0, 1)
    return image


def save_disparity(disparity, path, mult=1, equalize=False, defined=None, valid_range=None):
    disparity = mult * disparity.copy()
    if valid_range is not None:
        disparity = np.clip(disparity, valid_range[0], valid_range[1])
    disparity_min = np.min(disparity)
    disparity_max = np.max(disparity)
    if disparity_max == disparity_min:
        disparity_normalized = np.zeros_like(disparity, dtype='uint8')
    else:
        disparity_normalized = (disparity.astype(float) - disparity_min) * 255 / (disparity_max - disparity_min)
        disparity_normalized = np.clip(disparity_normalized, 0, 255).astype('uint8')
    cv2.imwrite(path, disparity_normalized)


def save_equalized_disparity(disparity, path, defined):
    disp = exposure.equalize_hist(disparity, mask=defined)
    disp[np.logical_not(defined)] = np.nan
    imsave(path, disp)


def open_tiff_file(path):
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Could not open {path}")
    img = np.array(ds.ReadAsArray())
    return img.copy()


def save_tiff_file(path, img):
    output_raster = gdal.GetDriverByName('GTiff').Create(
        path, img.shape[2], img.shape[1], img.shape[0], gdal.GDT_Float32
    )
    for i in range(img.shape[0]):
        band = output_raster.GetRasterBand(i + 1)
        band.WriteArray(img[i])


def imsave(path, image):
    image_min = np.nanmin(image)
    image_max = np.nanmax(image)
    if image_max == image_min:
        normalised_image = np.zeros_like(image)
    else:
        normalised_image = (image - image_min) * 255.0 / (image_max - image_min)
    normalised_image = np.nan_to_num(normalised_image, nan=0.0, posinf=255.0, neginf=0.0)
    normalised_image = np.clip(normalised_image, 0, 255).astype('uint8')

    image_color = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
    image_color[:,:,0] = normalised_image
    image_color[:,:,1] = normalised_image
    image_color[:,:,2] = normalised_image
    image_nan = np.isnan(image)
    image_color[image_nan, 0] = 255
    image_color[image_nan, 1] = 0
    image_color[image_nan, 2] = 0

    Image.fromarray(image_color).save(path)
