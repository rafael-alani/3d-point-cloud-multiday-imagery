import numpy as np
import cv2

class AffineCamera:
    # Affine camera approximation (eq 6)
    # using a least square solution solver (eq 7-8)
    
    def __init__(self, P, origin=None):
        self.P = P
        self.origin = origin 

    def from_rpc(self, rpc_meta, tile_bbox, h_range=(0, 50), origin=None):
        # convert from the rational function model, eq 1 to an affine camera model, eq 6
        c_min, r_min, w, h = tile_bbox
        
        # forward map the coordinates, eq 2
        center_c = c_min + w / 2.0
        center_r = r_min + h / 2.0
        center_h = np.mean(h_range)
        
        if origin is None:
            lat0, lon0 = rpc_meta.rpc_localization(center_r, center_c, center_h)
            origin = (lat0, lon0, center_h)
        else:
            lat0, lon0, _ = origin

        meters_per_deg_lat = 111132.0
        meters_per_deg_lon = 111132.0 * np.cos(np.radians(lat0))
        
        # generate probe points by sampling the forward mapping, eq 3-5
        cols = np.linspace(0, w, 3)
        rows = np.linspace(0, h, 3)
        heights = np.linspace(h_range[0], h_range[1], 2)
        
        obj_pts = []
        img_pts = []
        
        for r in rows:
            for c in cols:
                for z_h in heights:
                    lat, lon = rpc_meta.rpc_localization(r + r_min, c + c_min, z_h)
                    
                    X = (lon - lon0) * meters_per_deg_lon
                    Y = (lat - lat0) * meters_per_deg_lat
                    Z = z_h - origin[2]
                    
                    obj_pts.append([X, Y, Z])
                    img_pts.append([c, r])
                    
        obj_pts = np.array(obj_pts)
        img_pts = np.array(img_pts)
        
        # find the best linear fit (Affine) to the RPC projection 6-7-8
        X_aug = np.hstack([obj_pts, np.ones((obj_pts.shape[0], 1))])
        P_T, _, _, _ = np.linalg.lstsq(X_aug, img_pts, rcond=None)
        
        P = np.zeros((3, 4))
        P[0:2, :] = P_T.T
        P[2, 3] = 1.0
        
        return cls(P, origin)

    def project(self, X, Y, Z):
        uv = self.P @ np.array([X, Y, Z, 1.0])
        return u