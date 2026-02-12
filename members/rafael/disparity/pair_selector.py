
import os
from datetime import datetime
from dataclasses import dataclass
import itertools
import numpy as np
from .nitf_utils import NITFMetadata

@dataclass
class ImageCandidate:
    path: str
    filename: str
    date: datetime
    incidence_angle: float
    azimuth_angle: float
    view_vector: np.array
    
    cropped_path: str = None
    cropped_name: str = None
    
    bbox: tuple = None

    def __repr__(self):
        return f"<Img {self.filename}... Inc={self.incidence_angle:.1f}>"

@dataclass
class PairCandidate:
    img1: ImageCandidate
    img2: ImageCandidate
    convergence: float
    timediff: float
    
    rectified_pair_path: str = None   
    disparity_path: str = None

    ## MAYBE keep the mardcoedd
    rectified_name_L: str = None
    rectified_name_R: str = None
    disparity_name: str = None


class PairSelector:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.candidates = []

    def discover_images(self):

        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith(".ntf"):
                ntf_path = os.path.join(self.data_dir, filename)

                try:
                    meta = NITFMetadata(ntf_path)
                    view_vec = meta.get_view_vector()
                
                    cand = ImageCandidate(
                        path=ntf_path,
                        filename=filename,
                        date=meta.timestamp,
                        incidence_angle=meta.incidence_angle,
                        azimuth_angle=meta.azimuth_angle,
                        view_vector=view_vec
                    )
                    self.candidates.append(cand)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
        
        print(f"loaded {len(self.candidates)} nitf files")

    # Faciollo 2.1
    def select_pairs(self):
        valid_pairs = []
        late_bloomers = []
        
        # TODO: change to intertools quite slow without
        for c1, c2 in itertools.combinations(self.candidates, 2):

            # convergence angle of the 2 satellites
            # use dot product and then arccos to get the angle
            dot = np.dot(c1.view_vector, c2.view_vector)
            conv_angle = np.degrees(np.arccos(dot))

            timediff = abs((c1.date - c2.date).total_seconds())
            
             # as the disparity coloring was sometimes inverted,
            # keep track of orientation of the satellites relative to site
            sin_c1 = np.sin(np.radians(c1.azimuth_angle))
            sin_c2 = np.sin(np.radians(c2.azimuth_angle))
            if sin_c1 > sin_c2:
                c1, c2 = c2, c1

            if (not 5 <= conv_angle <= 45 or c1.incidence_angle > 40 or c2.incidence_angle > 40):
                late_bloomers.append(PairCandidate(c1, c2, conv_angle, timediff))
            else:
                valid_pairs.append(PairCandidate(c1, c2, conv_angle, timediff))

        print(f"found {len(valid_pairs)} valid pairs, found {len(late_bloomers)} late bloomers")
        return valid_pairs + late_bloomers