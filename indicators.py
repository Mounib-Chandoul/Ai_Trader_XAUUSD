import pandas as pd
import numpy as np

def calculate_ema(df, n=21, column='close'):
    """Exponential Moving Average as per Section 2."""
    return df[column].ewm(span=n, adjust=False).mean()

def calculate_atr(df, n=14):
    """Average True Range with Wilder's smoothing as per Section 2."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    return tr.ewm(alpha=1/n, adjust=False).mean()

def calculate_rsi(df, n=14):
    """Relative Strength Index with Wilder's smoothing as per Section 2."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD Line, Signal Line, and Histogram as per Section 2."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_hurst(series, window=100):
    """Hurst Exponent to detect regime (Trend vs Mean-Reversion)."""
    def hurst_calc(s):
        if len(s) < 20: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(s[lag:], s[:-lag]))) for lag in lags]
        # Filter out any non-positive values for log
        tau = [t if t > 0 else 1e-6 for t in tau]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    return series.rolling(window=window).apply(hurst_calc)

def calculate_adx(df, n=14):
    """Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0), index=df.index)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    tr_smoothed = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / tr_smoothed
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / tr_smoothed
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx

def calculate_kalman_filter(series):
    """Simple Kalman Filter for trend estimation as per Section 3."""
    n_iter = len(series)
    sz = (n_iter,)
    z = series.values
    
    Q = 1e-5 # process variance
    R = 0.1**2 # estimate of measurement variance
    
    xhat = np.zeros(sz)      # a posteri estimate of x
    P = np.zeros(sz)         # a posteri error estimate
    xhatminus = np.zeros(sz) # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain or blending factor
    
    xhat[0] = z[0]
    P[0] = 1.0
    
    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return pd.Series(xhat, index=series.index)

def detect_divergence(price_series, indicator_series, window=20):
    """Detects Bullish/Bearish Divergence as per Section 5."""
    if len(price_series) < window: return 0
    
    # Simple peak/trough detection
    p_last = price_series.iloc[-1]
    p_prev_peak = price_series.iloc[-window:-1].max()
    p_prev_trough = price_series.iloc[-window:-1].min()
    
    i_last = indicator_series.iloc[-1]
    i_prev_peak = indicator_series.iloc[-window:-1].max()
    i_prev_trough = indicator_series.iloc[-window:-1].min()
    
    # Bullish Divergence: Price lower low, Indicator higher low
    if p_last < p_prev_trough and i_last > i_prev_trough:
        return 1.0
    # Bearish Divergence: Price higher high, Indicator lower high
    if p_last > p_prev_peak and i_last < i_prev_peak:
        return -1.0
        
    return 0
