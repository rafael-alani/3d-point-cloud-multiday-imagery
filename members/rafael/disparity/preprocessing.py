from osgeo import gdal
import numpy as np  
from .pair_selector import ImageCandidate
import os

#https://gdal.org/en/stable/api/python/raster_api.html#osgeo.gdal.Dataset
def get_crop_area_from_kml(image: ImageCandidate, kml_path):
    kml_ds = gdal.OpenEx(kml_path)
    if not kml_ds:
        raise ValueError(f"ERROR - KML file not there: {kml_path}")
    
    # general function, for kml should only have 1 layer
    layer = kml_ds.GetLayer(0)
    min_lon, max_lon, min_lat, max_lat = layer.GetExtent()
    corners_geo = [
        (min_lon, max_lat),
        (max_lon, max_lat),
        (max_lon, min_lat),
        (min_lon, min_lat)
    ]

    # 2. Setup RPC Transformer (World -> Image Pixel)
    img_ds = gdal.Open(image.path)
    if not img_ds:
        raise ValueError(f"ERROR - NTF file not there: {image.path}")

    #https://gdal.org/en/stable/development/rfc/rfc22_rpc.html
    #https://gdal.org/en/stable/api/python/utilities.html
    transformer = gdal.Transformer(img_ds, None, ["METHOD=RPC"])
    pixel_coords = []
    for lon, lat in corners_geo:
        # Z=0 is sea level
        # by looking it on google earth, its only 17m..
        success, (col, row, z) = transformer.TransformPoint(1, lon, lat, 0)
        if not success:
            raise RuntimeError(f"ERROR - Could not transform coordinate ({lon}, {lat}) to pixels")
        pixel_coords.append((col, row))

    cols = [p[0] for p in pixel_coords]
    rows = [p[1] for p in pixel_coords]
    x_min = min(cols)
    x_max = max(cols)
    y_min = min(rows)
    y_max = max(rows)

    x_off = int(np.floor(x_min))
    y_off = int(np.floor(y_min))
    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))

    img_width = img_ds.RasterXSize
    img_height = img_ds.RasterYSize

    # edge cases
    if x_off < 0:
        width += x_off
        x_off = 0
    if y_off < 0:
        height += y_off
        y_off = 0
    if x_off + width > img_width:
        width = img_width - x_off
    if y_off + height > img_height:
        height = img_height - y_off

    return (x_off, y_off, width, height)

def generate_cropped(image: ImageCandidate, output_path, output_name, crop_area, crop_info=None):
    x_off, y_off, width, height = crop_area
    
    ds = gdal.Open(image.path)
    if ds is None:
        raise ValueError(f"ERROR - Could not open {image.path}")
    
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        srcWin=[x_off, y_off, width, height]
    )
    #output, datasource, options
    out_ds = gdal.Translate(os.path.join(output_path, output_name), ds, options=translate_options)
    
    if out_ds is None:
        raise ValueError(f"ERROR - Failed to crop image: {image.path}")
    
    # save crop_name before modifying for .npy
    tif_output_name = output_name
    
    # save crop info as .npy
    npy_name = output_name + '.npy'
    if crop_info is None:
        crop_info = np.array([x_off, y_off, x_off + width, y_off + height, 0, 0, 0, 0])
    np.save(os.path.join(output_path, npy_name), crop_info)

    image.cropped_path = output_path
    image.cropped_name = tif_output_name

