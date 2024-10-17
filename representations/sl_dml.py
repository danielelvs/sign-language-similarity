import numpy as np

from representations.base import BaseImageRepresentation


class SLDMLRepresentation(BaseImageRepresentation):
    name = "SL-DML"


    def transform(self, x, y, z):
        t = np.concatenate([x, y, z], axis=1)
        t -= np.min(t)
        t /= np.max(t)
        return t
