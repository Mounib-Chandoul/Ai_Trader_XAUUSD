import time
import pandas as pd
import numpy as np
from indicators import calculate_atr
from risk import RiskManager, calculate_lot_size, get_atr_stop_price
from signals import SignalGenerator
from ml_model import fetch_real_gold_data, MLModel

class GoldBot:
    def __init__(self, initial_balance=10000, risk_pc=1.0, ml_model=None):
        self.risk_manager = RiskManager(initial_balance)
        self.signal_gen = SignalGenerator(ml_model)
        self.risk_pc = risk_pc
        self.current_position = None
        self.trades = []
        # Track last trade for each session to allow one per session per day
        self.last_london_trade_date = None
        self.last_ny_trade_date = None
        
    def run_iteration(self, df):
        """One step of the bot using a window of data."""
        if self.risk_manager.check_circuit_breaker(self.risk_manager.balance):
            return

        current_time = df.index[-1]
        current_date = current_time.date()
        current_hour = current_time.hour
        current_price = df['close'].iloc[-1]
        atr = calculate_atr(df).iloc[-1]
        
        # Check for exit if position exists
        if self.current_position:
            pos = self.current_position
            side = pos['side']
            sl = pos['stop_loss']
            tp1 = pos['tp1']
            tp2 = pos['tp2']
            entry = pos['entry_price']
            
            # 1. Check for Stop Loss
            if (side == 'LONG' and current_price <= sl) or (side == 'SHORT' and current_price >= sl):
                self.close_position(current_price, current_hour)
                return

            # 2. Check for TP1 (Scale out 50% and move to Break Even)
            if not pos['tp1_hit']:
                if (side == 'LONG' and current_price >= tp1) or (side == 'SHORT' and current_price <= tp1):
                    print(f"[{current_date}] TP1 HIT: Scaling out 50% at {current_price:.2f} and moving SL to BE")
                    # Close 50% of the position
                    self.close_position(current_price, current_hour, portion=0.5)
                    # Remaining position is now Break Even
                    if self.current_position:
                        self.current_position['stop_loss'] = entry
                        self.current_position['tp1_hit'] = True
                    return

            # 3. Check for TP2
            if (side == 'LONG' and current_price >= tp2) or (side == 'SHORT' and current_price <= tp2):
                self.close_position(current_price, current_hour)
                return
            
            return

        # Define session windows (UTC)
        # London: 08:00 - 16:00, NY: 13:00 - 21:00
        LONDON_WINDOW = list(range(8, 17))
        NY_WINDOW = list(range(13, 22))
        
        is_london_start = (current_hour in LONDON_WINDOW)
        is_ny_start = (current_hour in NY_WINDOW)
        
        # Determine if we can trade in the current session
        session = None
        if is_london_start and self.last_london_trade_date != current_date:
            session = "LONDON"
        elif is_ny_start and self.last_ny_trade_date != current_date:
            session = "NY"
            
        if not session:
            return

        # Check for entry signal
        signal = self.signal_gen.get_signal(df)
        
        if signal in ['LONG', 'SHORT']:
            sl_price = get_atr_stop_price(signal, current_price, atr, k=2.0)
            risk_dist = abs(current_price - sl_price)
            
            # TP1 at 1.0x Risk, TP2 at 2.0x Risk
            tp1_price = current_price + risk_dist if signal == 'LONG' else current_price - risk_dist
            tp2_price = current_price + (2.0 * risk_dist) if signal == 'LONG' else current_price - (2.0 * risk_dist)
            
            # Volatility scaling
            returns = df['close'].pct_change().dropna()
            vol_mult = self.risk_manager.get_volatility_multiplier(returns)
            
            lot_size = calculate_lot_size(self.risk_manager.balance, self.risk_pc * vol_mult, risk_dist)
            
            if lot_size > 0:
                self.open_position(signal, current_price, sl_price, tp1_price, tp2_price, lot_size, current_date, session, current_hour)

    def open_position(self, side, price, sl, tp1, tp2, lots, date, session, hour):
        self.current_position = {
            'side': side,
            'entry_price': price,
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp1_hit': False,
            'lots': lots,
            'session': session,
            'entry_hour': hour
        }
        
        if session == "LONDON":
            self.last_london_trade_date = date
        else:
            self.last_ny_trade_date = date
            
        print(f"[{date} {session}] OPEN {side}: Price={price:.2f}, SL={sl:.2f}, TP1={tp1:.2f}, TP2={tp2:.2f}, Lots={lots}")

    def close_position(self, price, current_hour, portion=1.0):
        pos = self.current_position
        lots_to_close = pos['lots'] * portion
        
        pnl_per_oz = (price - pos['entry_price']) if pos['side'] == 'LONG' else (pos['entry_price'] - price)
        pnl = pnl_per_oz * lots_to_close * 100 # 100 oz per lot
        
        # Realistic dynamic spread simulation
        entry_hour = pos.get('entry_hour', 12)
        
        def get_spread(hour):
            if 13 <= hour <= 16: return 0.30 
            if (8 <= hour < 13) or (16 < hour <= 21): return 0.40 
            return 1.20 
            
        avg_spread = (get_spread(entry_hour) + get_spread(current_hour)) / 2
        costs = (avg_spread * lots_to_close * 100) + (0.70 * lots_to_close)
        final_pnl = pnl - costs
        
        self.risk_manager.update_balance(self.risk_manager.balance + final_pnl)
        self.trades.append(final_pnl)
        
        if portion == 1.0:
            print(f"CLOSE FULL: Price={price:.2f}, PnL=${final_pnl:.2f}, New Balance=${self.risk_manager.balance:.2f}")
            self.current_position = None
        else:
            print(f"CLOSE PARTIAL (50%): Price={price:.2f}, PnL=${final_pnl:.2f}")
            self.current_position['lots'] -= lots_to_close

def run_backtest():
    # 1. Fetch Real Data (15m granularity for last 60 days)
    data = fetch_real_gold_data(period="60d", interval="15m")
    
    # 2. Split into Train/Backtest (70/30)
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # 3. Train ML Model (with 4h horizon)
    print("\n--- Training Machine Learning Model (4h Horizon) ---")
    model = MLModel(horizon=16) 
    model.train(train_data)
    
    # 4. Run Simulation on Out-of-Sample data
    print("\n--- Starting Out-of-Sample Backtest (15m Precision) ---")
    bot = GoldBot(ml_model=model)
    
    # Lookback window for feature calculation (need ~200 bars for indicators)
    window_size = 200
    
    for i in range(window_size, len(test_data)):
        # Provide the bot with a window including the current bar
        window = test_data.iloc[i-window_size:i+1]
        bot.run_iteration(window)
        
    print(f"\nFinal Performance Metrics (Out-of-Sample):")
    print(f"Total Trades: {len(bot.trades)}")
    if bot.trades:
        win_rate = len([t for t in bot.trades if t > 0]) / len(bot.trades) * 100
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total PnL: ${sum(bot.trades):.2f}")
        print(f"Average PnL per Trade: ${sum(bot.trades)/len(bot.trades):.2f}")
    else:
        print("No trades executed.")

if __name__ == "__main__":
    run_backtest()
