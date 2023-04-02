import numpy as np
from sklearn.preprocessing import StandardScaler


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.
    """
    threshold = 1e-10

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale <= threshold:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale <= threshold] = 1.0
        return scale


class SafetyScaler(StandardScaler):
    """
    This is a scaler to eliminate negligible variance
    """

    def partial_fit(self, X, y=None, sample_weight=None):
        super().partial_fit(X, y, sample_weight)
        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        return np.nan_to_num(super().transform(X, copy), posinf=0, neginf=0)


if __name__ == '__main__':
    data = np.array([[0, 0], [0, 0], [1e-50, 1], [1, 1]])
    # data = np.random.randn(100, 100) * 100
    scaler = SafetyScaler()
    scaler.fit(data[:3, ])
    print(scaler.transform(data[3:, ]))
