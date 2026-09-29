"""Course 108: Autoregressive Distributed Lag (ARDL) & Granger Causality Model"""
import numpy as np

class ARDLGrangerCausality:
    @staticmethod
    def granger_f_stat(y: np.ndarray, x: np.ndarray, lag: int = 1) -> float:
        n = len(y) - lag
        y_t = y[lag:]
        y_lag = y[:-lag].reshape(-1, 1)
        x_lag = x[:-lag].reshape(-1, 1)
        
        # Restricted model (y on y_lag)
        xr = np.hstack([np.ones((n, 1)), y_lag])
        b_r = np.linalg.lstsq(xr, y_t, rcond=None)[0]
        ssr_r = np.sum((y_t - xr @ b_r)**2)
        
        # Unrestricted model (y on y_lag, x_lag)
        xu = np.hstack([np.ones((n, 1)), y_lag, x_lag])
        b_u = np.linalg.lstsq(xu, y_t, rcond=None)[0]
        ssr_u = np.sum((y_t - xu @ b_u)**2)
        
        f_stat = ((ssr_r - ssr_u) / lag) / (ssr_u / (n - 2 * lag - 1)) if ssr_u > 0 else 0.0
        return float(max(0.0, f_stat))
