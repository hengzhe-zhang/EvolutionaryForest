import numpy as np
from sklearn.preprocessing import StandardScaler

FLOAT32_MAX = np.finfo(np.float32).max


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.
    """
    threshold = 1e-10

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale <= threshold:
            scale = 1.0
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
        X_scaled = np.nan_to_num(super().transform(X, copy), posinf=0, neginf=0)
        if np.any(np.abs(X_scaled) > FLOAT32_MAX):
            centered = np.nan_to_num(X - getattr(self, "mean_", 0), posinf=0, neginf=0)
            return centered
        return X_scaled


if __name__ == "__main__":
    data = np.array([[0, 1], [0, 1], [1e-100, 1], [0, 1]]).astype(np.float32)
    # data = np.random.randn(100, 100) * 100
    scaler = SafetyScaler()
    scaler.fit(data)
    print(scaler.transform(data))
