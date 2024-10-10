import numpy as np
from typing import Tuple

from representations.base import BaseImageRepresentation


class SkeletonMagnitudeRepresentation(BaseImageRepresentation):
    name = "Skeleton-Magnitude"


    def __init__(self, temporal_scales=None):
        self.temporal_scales = temporal_scales if not None else [5, 10, 15]


    def transform(self, x, y, z):
        mag_values = []
        for t_scale in self.temporal_scales:
            diff_joint = np.array(self.compute_temporal_joint_difference(x, y, z, t_scale))
            mag_values.append(self.compute_joint_magnitude(diff_joint))

        img = np.array(mag_values)
        img = np.moveaxis(img, [0], [2])

        return img


    def compute_temporal_joint_difference(self, x, y, z, temporal_dist: int) -> Tuple[float, float, float]:
        shifted_x = np.roll(x, -temporal_dist)
        shifted_y = np.roll(y, -temporal_dist)
        shifted_z = np.roll(z, -temporal_dist)

        shifted_x[-temporal_dist:] = 0
        shifted_y[-temporal_dist:] = 0
        shifted_z[-temporal_dist:] = 0

        diff_x = x - shifted_x
        diff_y = y - shifted_x
        diff_z = z - shifted_x

        return diff_x, diff_y, diff_z


    def compute_joint_magnitude(self, diff_joint: np.array) -> float:
        ret = (diff_joint ** 2).sum(axis=0) ** (1. / 2)
        ret = self.normalize(ret, 0.0, 1, 1.0, 0.0)
        return ret


    def normalize(self, value: float, lower_bound: float, higher_bound: float, max_value: int, min_value: int) -> float:
        value[value > higher_bound] = max_value
        value[value < lower_bound] = min_value
        return value
