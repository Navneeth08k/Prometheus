'''
v10:
Big step back.

Rather than tryign to predict all the time, lets test out a simple rule based system
Based on how experts trade in market. 
Not expecting this to work incredibly well, but is a good stepping stone

Can mix this with ML approach.
'''

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator
from ta.momentum import ROCIndicator, RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Load and prepare data

def load_data(start_date='2000-01-01', end_date='2025-03-23'):
    # Fetch SPY and VIX data
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    
    # Resample to monthly
    spy_monthly = spy['Close'].resample('ME').last()  # Series, no to_frame()
    vix_monthly = vix['Close'].resample('ME').last().rename('VIX')
    
    # Calculate indicators
    sma50 = SMAIndicator(spy['Close'], window=50).sma_indicator().resample('ME').last()
    sma200 = SMAIndicator(spy['Close'], window=200).sma_indicator().resample('ME').last()
    roc21 = ROCIndicator(spy['Close'], window=21).roc().resample('ME').last()
    rsi14 = RSIIndicator(spy['Close'], window=14).rsi().resample('ME').last()
    bb = BollingerBands(spy['Close'], window=20)
    bb_high = bb.bollinger_hband().resample('ME').last()
    bb_low = bb.bollinger_lband().resample('ME').last()
    atr14 = AverageTrueRange(spy['High'], spy['Low'], spy['Close'], window=14).average_true_range().resample('ME').last()
    
    # Combine into DataFrame
    data = pd.concat([
        spy_monthly.rename('Close'), vix_monthly, sma50.rename('SMA50'), sma200.rename('SMA200'),
        roc21.rename('ROC21'), rsi14.rename('RSI'), bb_high.rename('BBHigh'),
        bb_low.rename('BBLow'), atr14.rename('ATR')
    ], axis=1).dropna()
    
    # Add 20-day high/low for breakouts
    data['High20'] = spy['Close'].rolling(20).max().resample('ME').last()
    data['Low20'] = spy['Close'].rolling(20).min().resample('ME').last()
    
    # Calculate monthly returns for stop-loss
    data['Returns'] = data['Close'].pct_change()
    
    return data

def calculate_thresholds(data, lookback=180):
    # Use daily data, resampled to monthly, over last 6 months (approx 180 days)
    window = data.tail(lookback)
    thresholds = {
        'vix_high': window['VIX'].quantile(0.80),
        'roc_positive': window['ROC21'].quantile(0.60),
        'roc_negative': window['ROC21'].quantile(0.40),
        'rsi_low': window['RSI'].quantile(0.20),
        'rsi_high': window['RSI'].quantile(0.80),
        'atr_breakout': window['ATR'].quantile(0.70)
    }
    return thresholds

# Trading decision tree
def trade(data, current_idx, thresholds, position):
    row = data.iloc[current_idx]
    signal = 'hold'  # Default
    
    # Risk Filter
    if row['VIX'] > thresholds['vix_high']:
        return 'sell' if position == 1 else 'hold'
    
    # Trend-Following
    if row['SMA50'] > row['SMA200'] and row['ROC21'] > thresholds['roc_positive']:
        signal = 'buy'
    elif row['SMA50'] < row['SMA200'] and row['ROC21'] < thresholds['roc_negative']:
        signal = 'sell'
    
    # Mean-Reversion (if no trend signal)
    if signal == 'hold':
        if row['RSI'] < thresholds['rsi_low'] and row['Close'] < row['BBLow']:
            signal = 'buy'
        elif row['RSI'] > thresholds['rsi_high'] and row['Close'] > row['BBHigh']:
            signal = 'sell'
    
    # Breakout (if no trend or reversion)
    if signal == 'hold':
        if row['Close'] > row['High20'] and row['ATR'] > thresholds['atr_breakout']:
            signal = 'buy'
        elif row['Close'] < row['Low20'] and row['ATR'] > thresholds['atr_breakout']:
            signal = 'sell'
    
    # Apply stop-loss if in position
    if position == 1 and row['Returns'] < -0.05:
        signal = 'sell'
    
    return signal

# Backtest the system
def backtest(data):
    portfolio_value = 10000.0
    position = 0  # 0 = cash, 1 = long
    portfolio = [portfolio_value]
    positions = [position]
    trades = []
    
    # Start after 6 months for initial thresholds
    start_idx = 6  # Approx 6 months of monthly data
    
    for i in range(start_idx, len(data)):
        # Update thresholds with prior 6 months
        lookback_data = data.iloc[max(0, i-6):i]
        thresholds = calculate_thresholds(lookback_data)
        
        # Get current price and signal
        current_price = data['Close'].iloc[i]
        signal = trade(data, i, thresholds, position)
        
        # Execute trade
        if signal == 'buy' and position == 0:
            position = 1
            trades.append(('buy', data.index[i], current_price))
        elif signal == 'sell' and position == 1:
            position = 0
            trades.append(('sell', data.index[i], current_price))
        
        # Update portfolio value
        if position == 1:
            portfolio_value = portfolio_value * (1 + data['Returns'].iloc[i])
        portfolio.append(portfolio_value)
        positions.append(position)
    
    # Convert to Series
    portfolio_series = pd.Series(portfolio, index=data.index[start_idx-1:])
    positions_series = pd.Series(positions, index=data.index[start_idx-1:])
    
    return portfolio_series, positions_series, trades

# Plot results
def plot_results(portfolio, trades, spy_data):
    spy_returns = spy_data['Close'].pct_change().add(1).cumprod() * 10000
    spy_portfolio = spy_returns.reindex(portfolio.index, method='ffill')
    
    plt.figure(figsize=(14, 8))
    plt.plot(portfolio, label=f'Prometheus v10 ({calculate_annual_return(portfolio):.2%}/year)', color='orange')
    plt.plot(spy_portfolio, label=f'SPY B&H ({calculate_annual_return(spy_portfolio):.2%}/year)', color='blue')
    
    # Plot trades
    buys = [t for t in trades if t[0] == 'buy']
    sells = [t for t in trades if t[0] == 'sell']
    if buys:
        buy_dates, buy_prices = zip(*[(t[1], t[2]) for t in buys])
        plt.scatter(buy_dates, portfolio.loc[buy_dates], marker='^', color='limegreen', label='Buy', s=100)
    if sells:
        sell_dates, sell_prices = zip(*[(t[1], t[2]) for t in sells])
        plt.scatter(sell_dates, portfolio.loc[sell_dates], marker='v', color='red', label='Sell', s=100)
    
    plt.title('Prometheus v10 vs SPY Buy-and-Hold ($10,000 Initial)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prometheus_v10_results.png', dpi=300)
    plt.show()

# Helper to calculate annualized return
def calculate_annual_return(portfolio):
    years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    total_return = (portfolio.iloc[-1] - portfolio.iloc[0]) / portfolio.iloc[0]
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

# Main execution
def main():
    print("Loading data...")
    data = load_data()
    print(f"Data loaded: {len(data)} monthly observations from {data.index[0]} to {data.index[-1]}")
    
    print("Running backtest...")
    portfolio, positions, trades = backtest(data)
    
    # Calculate metrics
    years = (data.index[-1] - data.index[0]).days / 365.25
    annual_return = calculate_annual_return(portfolio)
    spy_annual = calculate_annual_return(data['Close'].reindex(portfolio.index) * 10000 / data['Close'].iloc[0])
    win_rate = sum(1 for t in trades if t[0] == 'buy' and 
                   any(s[2] > t[2] for s in trades if s[0] == 'sell' and s[1] > t[1])) / len([t for t in trades if t[0] == 'buy']) if trades else 0
    max_drawdown = (portfolio / portfolio.cummax() - 1).min()
    
    print("\nPrometheus v10 Backtest Results:")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"SPY B&H Annual Return: {spy_annual:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Number of Trades: {len(trades)}")
    
    print("Generating plot...")
    plot_results(portfolio, trades, data)
    print("Plot saved as 'prometheus_v10_results.png'")

if __name__ == "__main__":
    main()