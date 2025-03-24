import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import timedelta


def load_data():
    """Load the saved feature matrix and models"""
    with open('feature_matrix.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    with open('indicators.pkl', 'rb') as f:
        indicators = pickle.load(f)
    with open('meta_model.pkl', 'rb') as f:
        meta_model = pickle.load(f)
    with open('ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)
    
    from tensorflow.keras.models import load_model
    try:
        lstm_model = load_model('lstm_model.h5')
    except Exception as e:
        print(f"LSTM model not found: {e}")
        lstm_model = None
        
    return feature_matrix, indicators, meta_model, ensemble_model, lstm_model

def create_sequences(X, timesteps=6):
    """Create sequences for LSTM input"""
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X.iloc[i:(i + timesteps)].values)
    return np.array(Xs)

def backtest(feature_matrix, indicators, meta_model, lstm_model, symbol='SPY', start_date=None, end_date=None):
    """
    Backtest with a $10,000 starting portfolio:
      - Position is either 0 (flat) or 1 (long). You can only sell once you've bought.
      - Only BUY if flat, only SELL if long.
      - Compare the strategy to a simple Buy & Hold (also starting at $10,000).
    """
    # Filter dates if provided
    if start_date:
        feature_matrix = feature_matrix[feature_matrix.index >= start_date]
        for key in indicators:
            indicators[key] = indicators[key][indicators[key].index >= start_date]
    if end_date:
        feature_matrix = feature_matrix[feature_matrix.index <= end_date]
        for key in indicators:
            indicators[key] = indicators[key][indicators[key].index <= end_date]
    
    # Download price data from yfinance
    price_data = yf.download(symbol, start=feature_matrix.index[0], end=feature_matrix.index[-1] + timedelta(30), progress=False)
    monthly_prices = price_data['Close'].resample('M').last()
    
    # Prepare monthly data for features, target and returns
    monthly_features = feature_matrix.resample('M').last()
    monthly_target = indicators[symbol]['target'].resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    # Align indices across all monthly data
    common_idx = monthly_features.index.intersection(monthly_target.index).intersection(monthly_returns.index)
    monthly_features = monthly_features.loc[common_idx]
    monthly_target = monthly_target.loc[common_idx]
    monthly_returns = monthly_returns.loc[common_idx]
    
    # Create LSTM sequences (for predictions only)
    X_lstm = create_sequences(feature_matrix, timesteps=6)
    
    # Prepare signals for the meta-model
    signals = [
        'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
        'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
        'inverse_cramer', 'momentum_21', 'mean_reversion_20'
    ]
    available_signals = [sig for sig in signals if sig in indicators[symbol].columns]
    missing_signals = set(signals) - set(available_signals)
    if missing_signals:
        print(f"Warning: Missing signals: {missing_signals}")
        for sig in missing_signals:
            indicators[symbol][sig] = 0
    
    meta_signals = indicators[symbol][signals].resample('M').last().reindex(common_idx).fillna(0)
    
    # Scale signals for meta-model
    scaler = StandardScaler()
    meta_signals_scaled = pd.DataFrame(scaler.fit_transform(meta_signals),
                                       index=meta_signals.index,
                                       columns=meta_signals.columns)
    
    # Get meta-model predictions
    try:
        meta_probs = meta_model.predict_proba(meta_signals_scaled)[:, 1]
        meta_preds = (meta_probs > 0.5).astype(int)
    except Exception as e:
        print(f"Meta prediction error: {e}")
        meta_preds = np.zeros(len(common_idx), dtype=int)
    
    # Get LSTM predictions (monthly)
    if lstm_model:
        try:
            lstm_probs = lstm_model.predict(X_lstm, verbose=0).flatten()
            lstm_series = pd.Series(lstm_probs, index=feature_matrix.index[6:]).resample('M').last()
            lstm_probs_monthly = lstm_series.reindex(common_idx).fillna(0)
            lstm_preds = (lstm_probs_monthly > 0.5).astype(int)
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            lstm_preds = np.zeros(len(common_idx), dtype=int)
    else:
        lstm_preds = np.zeros(len(common_idx), dtype=int)
    
    # Compute accuracies (against monthly_target)
    meta_accuracy = accuracy_score(monthly_target, meta_preds) if len(meta_preds) > 0 else 0
    lstm_accuracy = accuracy_score(monthly_target, lstm_preds) if (lstm_model and len(lstm_preds) > 0) else 0
    
    # --- Strategy Portfolio Updates ---
    # Start with $10,000
    meta_value = 10000.0
    lstm_value = 10000.0
    buy_hold_value = 10000.0
    
    meta_values = []
    lstm_values = []
    buy_hold_values = []
    
    # Positions: 0 = flat, 1 = long
    meta_position = 0
    lstm_position = 0
    
    # We'll record executed trades (for plotting) as a Series keyed by common_idx dates.
    meta_executed = pd.Series(index=common_idx, dtype=float)
    lstm_executed = pd.Series(index=common_idx, dtype=float)
    
    for date in common_idx:
        ret = monthly_returns.loc[date]
        
        # --- Meta strategy ---
        # Only buy if flat and signal is 1; only sell if long and signal is 0.
        idx = common_idx.get_loc(date)
        if meta_position == 0 and meta_preds[idx] == 1:
            meta_position = 1  # BUY
            meta_executed.loc[date] = 1
        elif meta_position == 1 and meta_preds[idx] == 0:
            meta_position = 0  # SELL
            meta_executed.loc[date] = 0
        else:
            meta_executed.loc[date] = np.nan  # No trade
        
        # Update meta portfolio value only if holding a position
        if meta_position == 1:
            meta_value *= (1 + ret)
        meta_values.append(meta_value)
        
        # --- LSTM strategy ---
        if lstm_position == 0 and lstm_preds[idx] == 1:
            lstm_position = 1  # BUY
            lstm_executed.loc[date] = 1
        elif lstm_position == 1 and lstm_preds[idx] == 0:
            lstm_position = 0  # SELL
            lstm_executed.loc[date] = 0
        else:
            lstm_executed.loc[date] = np.nan
        
        if lstm_position == 1:
            lstm_value *= (1 + ret)
        lstm_values.append(lstm_value)
        
        # --- Buy & Hold ---
        buy_hold_value *= (1 + ret)
        buy_hold_values.append(buy_hold_value)
    
    # Convert lists to Series
    meta_portfolio = pd.Series(meta_values, index=common_idx)
    lstm_portfolio = pd.Series(lstm_values, index=common_idx)
    buy_hold_portfolio = pd.Series(buy_hold_values, index=common_idx)
    
    # Compute cumulative returns relative to $10,000
    meta_cumulative = meta_portfolio / 10000.0 - 1
    lstm_cumulative = lstm_portfolio / 10000.0 - 1
    buy_hold_cumulative = buy_hold_portfolio / 10000.0 - 1
    
    # Compute annual returns (CAGR) if there is sufficient data
    if len(common_idx) > 1:
        years = (common_idx[-1] - common_idx[0]).days / 365.0
        buy_hold_annual = (1 + buy_hold_cumulative.iloc[-1]) ** (1 / years) - 1
        meta_annual = (1 + meta_cumulative.iloc[-1]) ** (1 / years) - 1
        lstm_annual = (1 + lstm_cumulative.iloc[-1]) ** (1 / years) - 1
    else:
        buy_hold_annual = meta_annual = lstm_annual = 0
    
    # Ensure final values and annual returns are floats (not Series)
    results = {
        'buy_hold_annual': float(buy_hold_annual),
        'meta_annual': float(meta_annual),
        'lstm_annual': float(lstm_annual),
        'meta_accuracy': meta_accuracy,
        'lstm_accuracy': lstm_accuracy,
        'buy_hold_cumulative': buy_hold_cumulative,
        'meta_cumulative': meta_cumulative,
        'lstm_cumulative': lstm_cumulative,
        'meta_executed': meta_executed,
        'lstm_executed': lstm_executed,
        'meta_final_value': float(meta_portfolio.iloc[-1]) if len(meta_portfolio) else 0,
        'lstm_final_value': float(lstm_portfolio.iloc[-1]) if len(lstm_portfolio) else 0,
        'buy_hold_final_value': float(buy_hold_portfolio.iloc[-1]) if len(buy_hold_portfolio) else 0
    }
    
    return results

def plot_results(results):
    """Plot the portfolio values and mark executed trades."""
    plt.figure(figsize=(12, 8))
    
    # Build absolute portfolio value series
    meta_portfolio = (results['meta_cumulative'] + 1) * 10000
    lstm_portfolio = (results['lstm_cumulative'] + 1) * 10000
    buy_hold_portfolio = (results['buy_hold_cumulative'] + 1) * 10000
    
    plt.plot(buy_hold_portfolio, label=f'Buy & Hold (${results["buy_hold_final_value"]:,.0f}; {results["buy_hold_annual"]:.2%}/yr)', color='blue')
    plt.plot(meta_portfolio, label=f'Meta (${results["meta_final_value"]:,.0f}; {results["meta_annual"]:.2%}/yr)', color='orange')
    plt.plot(lstm_portfolio, label=f'LSTM (${results["lstm_final_value"]:,.0f}; {results["lstm_annual"]:.2%}/yr)', color='green')
    
    # Mark executed trades for Meta
    meta_buy_idx = results['meta_executed'][results['meta_executed'] == 1].index
    meta_sell_idx = results['meta_executed'][results['meta_executed'] == 0].index
    plt.scatter(meta_buy_idx, meta_portfolio.reindex(meta_buy_idx), marker='^', color='limegreen', s=100, label='Meta Buy', zorder=5)
    plt.scatter(meta_sell_idx, meta_portfolio.reindex(meta_sell_idx), marker='v', color='red', s=100, label='Meta Sell', zorder=5)
    
    # Mark executed trades for LSTM
    lstm_buy_idx = results['lstm_executed'][results['lstm_executed'] == 1].index
    lstm_sell_idx = results['lstm_executed'][results['lstm_executed'] == 0].index
    plt.scatter(lstm_buy_idx, lstm_portfolio.reindex(lstm_buy_idx), marker='^', color='darkgreen', s=100, label='LSTM Buy', zorder=5)
    plt.scatter(lstm_sell_idx, lstm_portfolio.reindex(lstm_sell_idx), marker='v', color='darkred', s=100, label='LSTM Sell', zorder=5)
    
    plt.title("Backtest Comparison: $10,000 Starting Capital\n(Buy & Hold vs. Meta vs. LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('backtest_comparison.png', dpi=300)
    plt.show()

def main():
    feature_matrix, indicators, meta_model, ensemble_model, lstm_model = load_data()
    symbol = list(indicators.keys())[0]
    print(f"Using symbol: {symbol}")
    
    results = backtest(feature_matrix, indicators, meta_model, lstm_model, symbol=symbol)
    
    print("Backtest Results:")
    print(f"Buy & Hold Final Value: ${results['buy_hold_final_value']:,.2f} (Annual Return: {results['buy_hold_annual']:.2%})")
    print(f"Meta Final Value:      ${results['meta_final_value']:,.2f} (Annual Return: {results['meta_annual']:.2%})")
    print(f"LSTM Final Value:      ${results['lstm_final_value']:,.2f} (Annual Return: {results['lstm_annual']:.2%})")
    print(f"Meta Model Accuracy vs. Monthly Target: {results['meta_accuracy']:.4f}")
    print(f"LSTM Model Accuracy vs. Monthly Target: {results['lstm_accuracy']:.4f}")
    
    plot_results(results)
    
    # Export cumulative returns and drawdowns for further analysis
    df = pd.DataFrame({
        'Buy_Hold': results['buy_hold_cumulative'],
        'Meta': results['meta_cumulative'],
        'LSTM': results['lstm_cumulative']
    })
    df.to_csv('backtest_cumulative_returns.csv')
    
    drawdowns = df.apply(lambda x: x - x.cummax())
    drawdowns.to_csv('backtest_drawdowns.csv')
    
    print("\nMaximum Drawdowns:")
    for col in df.columns:
        dd_min = (df[col] - df[col].cummax()).min()
        print(f"{col} Drawdown: {dd_min:.2%}")

if __name__ == "__main__":
    main()
