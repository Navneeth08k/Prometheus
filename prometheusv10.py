import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import timedelta
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import warnings

# Suppress FutureWarnings from stable-baselines3
warnings.filterwarnings("ignore", category=FutureWarning)

# Custom Gymnasium Environment for Trading
class TradingEnv(Env):
    def __init__(self, feature_matrix, indicators, monthly_returns, symbol='SPY'):
        super(TradingEnv, self).__init__()
        self.feature_matrix = feature_matrix
        self.indicators = indicators
        self.monthly_returns = monthly_returns
        self.symbol = symbol
        self.signals = [
            'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
            'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
            'inverse_cramer', 'momentum_21', 'mean_reversion_20'
        ]
        self.action_space = Discrete(2)  # 0 = sell/flat, 1 = buy/long
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(self.signals),), dtype=np.float32)
        self.scaler = StandardScaler()
        self.seed_value = None
        self.max_steps = len(monthly_returns)  # Limit to dataset length
        self.reset()

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.position = 0  # 0 = flat, 1 = long
        self.portfolio_value = 10000.0  # Starting with $10,000
        self.common_idx = self.monthly_returns.index
        self.meta_signals = self.indicators[self.symbol][self.signals].resample('ME').last().reindex(self.common_idx).fillna(0)
        self.meta_signals_scaled = pd.DataFrame(self.scaler.fit_transform(self.meta_signals),
                                                index=self.meta_signals.index,
                                                columns=self.meta_signals.columns)
        return self._get_observation(), {}

    def step(self, action):
        reward = 0
        if self.position == 0 and action == 1:  # Buy
            self.position = 1
        elif self.position == 1 and action == 0:  # Sell
            self.position = 0

        if self.position == 1:
            reward = float(self.monthly_returns.iloc[self.current_step])  # Ensure scalar
            self.portfolio_value *= (1 + reward)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps - 1
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape)
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        return self.meta_signals_scaled.iloc[self.current_step].values

def load_data():
    with open('feature_matrix.pkl', 'rb') as f:
        feature_matrix = pickle.load(f)
    with open('indicators.pkl', 'rb') as f:
        indicators = pickle.load(f)
    with open('ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)
    from tensorflow.keras.models import load_model
    try:
        lstm_model = load_model('lstm_model.h5')
    except Exception as e:
        print(f"LSTM model not found: {e}")
        lstm_model = None
    return feature_matrix, indicators, ensemble_model, lstm_model

def create_sequences(X, timesteps=6):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X.iloc[i:(i + timesteps)].values)
    return np.array(Xs)

def backtest_rl(feature_matrix, indicators, lstm_model, symbol='SPY'):
    # Fetch price data
    price_data = yf.download(symbol, start=feature_matrix.index[0], end=feature_matrix.index[-1] + timedelta(30), progress=False)
    monthly_prices = price_data['Close'].resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    monthly_target = indicators[symbol]['target'].resample('ME').last()

    # Align indices
    common_idx = monthly_returns.index.intersection(monthly_target.index)
    monthly_returns = monthly_returns.loc[common_idx]
    monthly_target = monthly_target.loc[common_idx]

    # LSTM predictions (unchanged)
    X_lstm = create_sequences(feature_matrix, timesteps=6)
    if lstm_model:
        lstm_probs = lstm_model.predict(X_lstm, verbose=0).flatten()
        lstm_series = pd.Series(lstm_probs, index=feature_matrix.index[6:]).resample('ME').last()
        lstm_probs_monthly = lstm_series.reindex(common_idx).fillna(0)
        lstm_preds = (lstm_probs_monthly > 0.5).astype(int)
    else:
        lstm_preds = np.ones(len(common_idx), dtype=int)

    # RL Environment and Training
    env = make_vec_env(lambda: TradingEnv(feature_matrix, indicators, monthly_returns, symbol), n_envs=1)
    total_steps = min(1000, len(monthly_returns) * 2)  # Cap at 2x dataset length or 1000
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')  # Verbose for progress
    print(f"Training PPO for {total_steps} timesteps...")
    model.learn(total_timesteps=total_steps)

    # Backtest RL Policy
    obs = env.reset()  # Single return value
    rl_portfolio = [10000.0]
    rl_positions = []
    rl_executed = []
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)  # Unpack 4 values
        rl_portfolio.append(env.envs[0].portfolio_value)
        rl_positions.append(env.envs[0].position)
        rl_executed.append(action if action != (rl_positions[-2] if len(rl_positions) > 1 else -1) else np.nan)

    rl_cumulative = pd.Series(rl_portfolio[1:], index=common_idx) - 10000.0
    rl_preds = pd.Series(rl_executed, index=common_idx)
    rl_positions = pd.Series(rl_positions, index=common_idx)

    # Buy-and-Hold Benchmarks
    benchmarks = {'SPY': monthly_returns}
    for ticker in ['QQQ', 'DIA', 'IWM']:
        bench_data = yf.download(ticker, start=feature_matrix.index[0], end=feature_matrix.index[-1] + timedelta(30), progress=False)
        bench_prices = bench_data['Close'].resample('ME').last()
        bench_returns = bench_prices.pct_change().dropna().reindex(common_idx).fillna(0)
        benchmarks[ticker] = bench_returns

    bh_portfolios = {ticker: [10000.0] for ticker in benchmarks}
    for ticker, returns in benchmarks.items():
        for ret in returns:
            bh_portfolios[ticker].append(bh_portfolios[ticker][-1] * (1 + ret))
        bh_portfolios[ticker] = pd.Series(bh_portfolios[ticker][1:], index=common_idx) - 10000.0

    # Metrics
    years = (common_idx[-1] - common_idx[0]).days / 365.0
    results = {
        'rl_cumulative': rl_cumulative,
        'rl_preds': rl_preds,
        'rl_positions': rl_positions,
        'rl_annual': (1 + rl_cumulative.iloc[-1] / 10000.0) ** (1/years) - 1,
        'rl_accuracy': accuracy_score(monthly_target, (rl_preds.dropna() == 1).astype(int)),
        'lstm_accuracy': accuracy_score(monthly_target, lstm_preds)
    }
    for ticker in benchmarks:
        results[f'{ticker}_cumulative'] = bh_portfolios[ticker]
        results[f'{ticker}_annual'] = (1 + bh_portfolios[ticker].iloc[-1] / 10000.0) ** (1/years) - 1

    # Debug
    print(f"RL portfolio (first 20): {rl_portfolio[:20]}")
    print(f"RL positions (first 20): {rl_positions[:20].tolist()}")
    print(f"RL executed (first 20): {rl_executed[:20]}")
    print(f"Monthly returns (first 10): {monthly_returns[:10].tolist()}")

    return results

def plot_results(results):
    plt.figure(figsize=(14, 10))
    plt.plot(results['rl_cumulative'], label=f'RL Meta ({results["rl_annual"]:.2%}/year)', color='orange')
    for ticker in ['SPY', 'QQQ', 'DIA', 'IWM']:
        plt.plot(results[f'{ticker}_cumulative'], label=f'{ticker} B&H ({results[f"{ticker}_annual"]:.2%}/year)')

    rl_buy = results['rl_cumulative'][results['rl_preds'] == 1]
    rl_sell = results['rl_cumulative'][results['rl_preds'] == 0]
    plt.scatter(rl_buy.index, rl_buy, marker='^', color='limegreen', label='RL Buy', s=100, zorder=5)
    plt.scatter(rl_sell.index, rl_sell, marker='v', color='red', label='RL Sell', s=100, zorder=5)

    plt.title('RL Meta Model vs Buy-and-Hold ($10,000 Initial)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value Change ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('backtest_results_v10.png', dpi=300)
    plt.show()

def main():
    feature_matrix, indicators, ensemble_model, lstm_model = load_data()
    symbol = list(indicators.keys())[0]
    print(f"Using symbol: {symbol}")

    results = backtest_rl(feature_matrix, indicators, lstm_model, symbol=symbol)

    print("Backtest Results:")
    print(f"RL Meta Annual Return: {results['rl_annual']:.2%}")
    for ticker in ['SPY', 'QQQ', 'DIA', 'IWM']:
        print(f"{ticker} Buy & Hold Annual Return: {results[f'{ticker}_annual']:.2%}")
    print(f"RL Meta Accuracy: {results['rl_accuracy']:.4f}")
    print(f"LSTM Model Accuracy: {results['lstm_accuracy']:.4f}")

    plot_results(results)

if __name__ == "__main__":
    main()