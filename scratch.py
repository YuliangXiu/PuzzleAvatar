import math
import random

import torch

class Solution:
    def __init__(self) -> None:
        self.elevation_range = [0, 30]
        self.azimuth_range = [0, 360]
        self.n_view = 4

    def run(self, real_batch_size=2):
        # sample elevation angles
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            # same elevation for all views in a batch
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0
        
        print(elevation_deg.shape)
        # sample azimuth angles from a uniform distribution bounded by azimuth_range

        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1,1) + torch.arange(self.n_view).reshape(1,-1)
        ).reshape(-1) / self.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180
        print(azimuth_deg.shape, azimuth_deg)

if __name__ == "__main__":
    sol = Solution()
    sol.run()

