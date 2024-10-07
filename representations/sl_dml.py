import numpy as np

from representations.base import BaseRepresentation


class SLDMLRepresentation(BaseRepresentation):
    name = "SL-DML"

    def transform(self, x, y, z):
        t = np.concatenate([x, y, z], axis=1)
        t -= np.min(t)
        t /= np.max(t)
        return t
