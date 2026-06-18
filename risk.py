import math

def calculate_lot_size(balance, risk_pc, stop_distance, oz_per_lot=100):
    """
    Calculates lot size based on balance risk and ATR stop distance.
    As per Section 7 and 4.
    """
    if stop_distance <= 0:
        return 0
    
    risk_amount = balance * (risk_pc / 100)
    # 1 oz of gold moves $1 for every $1 price move.
    # Risk_Amount = Lot_Size * oz_per_lot * stop_distance
    lot_size = risk_amount / (stop_distance * oz_per_lot)
    
    # Round down to nearest micro lot (0.01)
    return math.floor(lot_size * 100) / 100

def get_atr_stop_price(side, entry_price, atr, k=2.0):
    """
    Calculates stop loss price based on ATR.
    As per Section 4.
    """
    if side == 'LONG':
        return entry_price - (k * atr)
    else:
        return entry_price + (k * atr)

class RiskManager:
    def __init__(self, initial_balance, daily_loss_limit_pc=2.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.daily_loss_limit_pc = daily_loss_limit_pc
        self.start_day_balance = initial_balance
        
    def check_circuit_breaker(self, current_balance):
        """Halt if daily drawdown > 2% as per Section 4."""
        drawdown = (self.start_day_balance - current_balance) / self.start_day_balance
        if drawdown >= (self.daily_loss_limit_pc / 100):
            return True # Circuit breaker triggered
        return False
    
    def get_volatility_multiplier(self, returns_series, window=30):
        """
        Halves position size during high volatility regimes as per Section 3/8.
        """
        if len(returns_series) < window:
            return 1.0
        
        current_vol = returns_series.tail(1).std()
        avg_vol = returns_series.tail(window).std()
        
        if current_vol > (1.5 * avg_vol):
            return 0.5
        return 1.0

    def update_balance(self, new_balance):
        self.balance = new_balance
