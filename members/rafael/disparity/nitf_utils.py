
import rasterio
import numpy as np
from datetime import datetime

class NITFMetadata:
    def __init__(self, filepath):
        self.filepath = filepath
        self.incidence_angle = None
        self.azimuth_angle = None
        self.timestamp = None

        self.rpcs: rasterio.rpc.RPC
        
        self.parse_metadata()

    def parse_metadata(self):
        try:
            with rasterio.open(self.filepath) as src:
                tags = src.tags()

                obliquity = float(tags.get('NITF_USE00A_OBL_ANG'))
                # by printing the worst and best incidence angle it seems OBL_ANG is incidence
                # not 90 - obliquity
                self.incidence_angle = obliquity

                # azimuth angle - measure from north
                # orientation of the satellite when taking the image
                self.azimuth_angle = float(tags.get("NITF_CSEXRA_AZ_OF_OBLIQUITY"))
                
                date_str = tags.get("NITF_STDIDC_ACQUISITION_DATE")
                self.timestamp = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                
                self.rpcs = src.rpcs
                
        except Exception as e:
            print(f"Error reading metadata from {self.filepath}: {e}")

    def get_view_vector(self):
        """
        Calculate the view vector in Cartesian coordinates from Spherical (Incidence, Azimuth).
        
        Once again:
        Incidence is off-nadir angle (0 = nadir, 90 = horizon) 
        Azimuth is clockwise from North.
        
        To compare the 2 satellites views of the same site we need a common coordinate system
        Use a local coordinate system, East North, Up (ENU):

        Z = Up
        Y = North
        X = East
        """

        inc_rad = np.radians(self.incidence_angle)
        az_rad = np.radians(self.azimuth_angle)
        
        # Spherical to Cartesian
        # View vector pointing FROM ground TO satellite or vice-versa. 
        # For convergence angle, the direction relative to vertical matters.
        # Let's define vector pointing TO satellite.
        # Horizontal component is sin(Incidence)
        
        # think of the triangle formed between the satellite, the ground and the point on the normal of hte ground that
        # makes a 90 deg triangle
        z = np.cos(inc_rad)
        h = np.sin(inc_rad)
        
        
        # to wrap your head around azimuth think if you project hte normal of the ground on the plane that is parallel
        # to the ground (it would be a point C)
        # and you would ask where can i place my satelite S so that i have the incidence angle I
        # you would form a circle of possible positions for S with center C and radius h

        # for the azimuth angle gives you the exact placement on this circle
        # 0deg is north, 90deg is east, 180deg is south, 270deg is west
        # think of th cartesian coordinates rotated 90deg counterclockwise
        y = h * np.cos(az_rad) # North component
        x = h * np.sin(az_rad) # East component
        
        return np.array([x, y, z])