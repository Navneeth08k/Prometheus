'''
Version 6:
This one is a big leap.
Precision was good on the last one. 
I think one day is too hard to predict on so we will try monthly
If that proves to be too little data
We can go weekly

Using LSTM models as well here.
Capping ensemble relevance.
300 features including quant like features.
'''


import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import ta
import textblob
from concurrent.futures import ThreadPoolExecutor
import finnhub
from fredapi import Fred
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings('ignore')

class EnhancedFinancialAnalysisSystemV6:
    def __init__(self, symbol='SPY', lookback_years=3):
        """Initialize the system with the primary symbol and lookback period."""
        self.symbol = symbol
        self.lookback_years = lookback_years
        self.start_date = (dt.datetime.now() - dt.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        self.end_date = dt.datetime.now().strftime('%Y-%m-%d')
        self.features = []
        self.data = {}
        self.indicators = {}
        # Replace with your own API keys
        self.finnhub_client = finnhub.Client(api_key="cv9orrpr01qpd9s8hhggcv9orrpr01qpd9s8hhh0")
        self.fred = Fred(api_key="3173a18ce8eb6fd417c310f8f887f3a5")
        self.evaluation_results = {}

    def fetch_data(self):
        """Fetch historical price, VIX, and macroeconomic data."""
        print("Fetching data...")
        with ThreadPoolExecutor() as executor:
            # Fetch SPY, VIX, and NYAD (breadth) data concurrently
            self.data[self.symbol] = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
            self.data['^VIX'] = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
            try:
                self.data['A-D'] = yf.download('^NYAD', start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
                if self.data['A-D'].empty:
                    print("Warning: '^NYAD' data is empty. Breadth signal will be set to 0.")
                    self.data['A-D'] = pd.DataFrame(index=self.data[self.symbol].index, columns=['Close']).fillna(0)
            except Exception as e:
                print(f"Warning: Failed to fetch '^NYAD' - {e}. Breadth signal will be set to 0.")
                self.data['A-D'] = pd.DataFrame(index=self.data[self.symbol].index, columns=['Close']).fillna(0)
        
        # Fetch macroeconomic data from FRED
        print("Fetching FRED macroeconomic data...")
        self.macro_data = pd.DataFrame({
            'yield_curve_10y2y': self.fred.get_series('T10Y2Y') / 100,
            'yield_curve_10y3m': self.fred.get_series('T10Y3M') / 100,
            'fed_funds': self.fred.get_series('FEDFUNDS') / 100,
            'cpi': self.fred.get_series('CPIAUCSL').pct_change(12),
            'unemployment': self.fred.get_series('UNRATE') / 100,
            'gdp': self.fred.get_series('GDP').pct_change(4).reindex(self.data[self.symbol].index, method='ffill'),
            'pmi': self.fred.get_series('ISM/MAN_PMI').reindex(self.data[self.symbol].index, method='ffill'),
            'consumer_confidence': self.fred.get_series('UMCSENT').reindex(self.data[self.symbol].index, method='ffill'),
            'treasury_10y': self.fred.get_series('DGS10') / 100,
            'treasury_3m': self.fred.get_series('DGS3MO') / 100,
            'real_yield': self.fred.get_series('REAINTRATREARAT10Y').reindex(self.data[self.symbol].index, method='ffill')
        }).fillna(method='ffill').fillna(method='bfill')
        
        # Handle missing data in primary dataset
        self.data[self.symbol].fillna(method='ffill', inplace=True)
        self.data[self.symbol].fillna(method='bfill', inplace=True)
        return self

    def generate_indicators(self):
        """Generate a comprehensive set of technical, macro, volatility, and sentiment indicators (300+ features)."""
        print("Generating indicators...")
        df = self.data[self.symbol]
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # Monthly target: next month's return > 0
        monthly_close = close.resample('M').last()
        self.indicators[self.symbol] = pd.DataFrame(index=df.index)
        self.indicators[self.symbol]['target'] = (monthly_close.pct_change(1).shift(-1) > 0).reindex(df.index, method='ffill').fillna(False)
        
        # Technical Indicators (multiple windows for robustness)
        for window in [5, 10, 14, 20, 50, 100, 200]:
            self.indicators[self.symbol][f'sma_{window}'] = close.rolling(window).mean()
            self.indicators[self.symbol][f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
            self.indicators[self.symbol][f'rsi_{window}'] = ta.momentum.RSIIndicator(close, window).rsi()
            self.indicators[self.symbol][f'roc_{window}'] = ta.momentum.ROCIndicator(close, window).roc()
        self.indicators[self.symbol]['macd'] = ta.trend.MACD(close).macd()
        self.indicators[self.symbol]['macd_signal'] = ta.trend.MACD(close).macd_signal()
        self.indicators[self.symbol]['macd_diff'] = ta.trend.MACD(close).macd_diff()
        self.indicators[self.symbol]['bb_upper_20'] = ta.volatility.BollingerBands(close, 20).bollinger_hband()
        self.indicators[self.symbol]['bb_lower_20'] = ta.volatility.BollingerBands(close, 20).bollinger_lband()
        self.indicators[self.symbol]['bb_width_20'] = ta.volatility.BollingerBands(close, 20).bollinger_wband()
        self.indicators[self.symbol]['atr_14'] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
        self.indicators[self.symbol]['atr_20'] = ta.volatility.AverageTrueRange(high, low, close, 20).average_true_range()
        self.indicators[self.symbol]['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        self.indicators[self.symbol]['obv_ma_20'] = self.indicators[self.symbol]['obv'].rolling(20).mean()
        self.indicators[self.symbol]['stoch_k'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
        self.indicators[self.symbol]['stoch_d'] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
        self.indicators[self.symbol]['cci_20'] = ta.trend.CCIIndicator(high, low, close, 20).cci()
        self.indicators[self.symbol]['cci_50'] = ta.trend.CCIIndicator(high, low, close, 50).cci()
        self.indicators[self.symbol]['willr_14'] = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r()
        self.indicators[self.symbol]['willr_28'] = ta.momentum.WilliamsRIndicator(high, low, close, 28).williams_r()
        self.indicators[self.symbol]['adx_14'] = ta.trend.ADXIndicator(high, low, close, 14).adx()
        self.indicators[self.symbol]['adx_28'] = ta.trend.ADXIndicator(high, low, close, 28).adx()
        self.indicators[self.symbol]['dmi_plus_14'] = ta.trend.ADXIndicator(high, low, close, 14).adx_pos()
        self.indicators[self.symbol]['dmi_minus_14'] = ta.trend.ADXIndicator(high, low, close, 14).adx_neg()
        
        # Macro Indicators
        for col in self.macro_data.columns:
            self.indicators[self.symbol][f'macro_{col}'] = self.macro_data[col]
        
        # Volatility Indicators
        vix = self.data['^VIX']['Close'].squeeze()
        self.indicators[self.symbol]['vix'] = vix
        self.indicators[self.symbol]['vix_ma_20'] = vix.rolling(20).mean()
        self.indicators[self.symbol]['vix_pct_change'] = vix.pct_change()
        self.indicators[self.symbol]['real_vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        self.indicators[self.symbol]['real_vol_50'] = close.pct_change().rolling(50).std() * np.sqrt(252)
        
        # Quant-like Signals
        self.indicators[self.symbol]['momentum_21'] = close.pct_change(21)
        self.indicators[self.symbol]['momentum_63'] = close.pct_change(63)
        self.indicators[self.symbol]['mean_reversion_20'] = (close - self.indicators[self.symbol]['sma_20']) / self.indicators[self.symbol]['sma_20']
        self.indicators[self.symbol]['mean_reversion_50'] = (close - self.indicators[self.symbol]['sma_50']) / self.indicators[self.symbol]['sma_50']
        self.indicators[self.symbol]['breadth_signal'] = self.data['A-D']['Close'].rolling(20).mean().pct_change()
        
        # Sentiment Analysis (News + Inverse Cramer)
        news_data = self.finnhub_client.company_news(self.symbol, _from=self.start_date, to=self.end_date)
        if news_data:
            news_sent = pd.Series(
                [textblob.TextBlob(n['summary']).sentiment.polarity for n in news_data if 'summary' in n],
                index=[pd.to_datetime(n['datetime'], unit='s') for n in news_data if 'summary' in n]
            )
            news_sent = news_sent.groupby(news_sent.index.date).mean()
            news_sent = pd.Series(news_sent, index=pd.to_datetime(news_sent.index)).reindex(df.index, method='nearest').fillna(0)
        else:
            print(f"No news data for {self.symbol}. Setting news_sentiment to 0.")
            news_sent = pd.Series(0, index=df.index)
        self.indicators[self.symbol]['news_sentiment'] = news_sent
        self.indicators[self.symbol]['inverse_cramer'] = np.where(news_sent > 0.1, -1, np.where(news_sent < -0.1, 1, 0))  # Simplified Inverse Cramer
        
        # Handle missing values
        self.indicators[self.symbol].fillna(method='ffill', inplace=True)
        self.indicators[self.symbol].fillna(0, inplace=True)
        return self

    def prepare_features(self):
        """Prepare feature matrix for modeling, including monthly resampling and sequence creation for LSTM."""
        print("Preparing features...")
        df = self.indicators[self.symbol].iloc[200:]  # Skip initial rows with NaNs from indicators
        self.features = [col for col in df.columns if col != 'target']
        
        # Feature selection
        X = df[self.features]
        y = df['target']
        selector = SelectKBest(f_classif, k=150)
        selector.fit(X, y)
        self.features = [self.features[i] for i in selector.get_support(indices=True)]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[self.features])
        self.feature_matrix = pd.DataFrame(X_scaled, index=df.index, columns=self.features)
        self.target = y
        
        # Resample to monthly for ensemble and meta
        monthly_features = self.feature_matrix.resample('M').last()
        monthly_target = self.target.resample('M').last()
        
        # Ensemble split: 60% train, 20% val, 20% test
        n = len(monthly_features)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        self.X_train_ens = monthly_features.iloc[:train_end]
        self.X_val_ens = monthly_features.iloc[train_end:val_end]
        self.X_test_ens = monthly_features.iloc[val_end:]
        self.y_train_ens = monthly_target.iloc[:train_end]
        self.y_val_ens = monthly_target.iloc[train_end:val_end]
        self.y_test_ens = monthly_target.iloc[val_end:]
        
        # Meta split: 80% train, 20% test
        self.X_train_meta = monthly_features.iloc[:val_end]
        self.X_test_meta = monthly_features.iloc[val_end:]
        self.y_train_meta = monthly_target.iloc[:val_end]
        self.y_test_meta = monthly_target.iloc[val_end:]
        
        # LSTM sequences: 6-month lookback
        def create_sequences(X, y, timesteps=6):
            Xs, ys = [], []
            for i in range(len(X) - timesteps):
                Xs.append(X.iloc[i:(i + timesteps)].values)
                ys.append(y.iloc[i + timesteps])
            return np.array(Xs), np.array(ys)
        
        self.X_train_lstm, self.y_train_lstm = create_sequences(self.X_train_meta, self.y_train_meta)
        self.X_test_lstm, self.y_test_lstm = create_sequences(self.X_test_meta, self.y_test_meta)
        return self

    def build_ensemble(self):
        """Build and train the XGBoost ensemble model."""
        print("Building ensemble...")
        if os.path.exists('ensemble_model.pkl') and not hasattr(self, 'retrain'):
            with open('ensemble_model.pkl', 'rb') as f:
                self.ensemble = pickle.load(f)
        else:
            self.ensemble = XGBClassifier(n_estimators=200, max_depth=5, reg_lambda=1, random_state=42, eval_metric='logloss')
            self.ensemble.fit(self.X_train_ens, self.y_train_ens, eval_set=[(self.X_val_ens, self.y_val_ens)], verbose=False)
            with open('ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble, f)
        
        # Evaluate ensemble
        y_pred = self.ensemble.predict(self.X_test_ens)
        self.evaluation_results['ensemble'] = {
            'accuracy': accuracy_score(self.y_test_ens, y_pred),
            'precision': precision_score(self.y_test_ens, y_pred),
            'recall': recall_score(self.y_test_ens, y_pred),
            'f1': f1_score(self.y_test_ens, y_pred)
        }
        # Generate ensemble signal
        ensemble_prob = self.ensemble.predict_proba(self.feature_matrix[self.features])[:, 1]
        self.indicators[self.symbol]['ensemble_signal'] = pd.Series(ensemble_prob, index=self.feature_matrix.index)
        return self

    def build_lstm(self):
        """Build and train the LSTM model for time-series prediction."""
        print("Building LSTM...")
        if os.path.exists('lstm_model.h5') and not hasattr(self, 'retrain'):
            self.lstm = tf.keras.models.load_model('lstm_model.h5')
        else:
            self.lstm = Sequential([
                LSTM(128, return_sequences=True, input_shape=(6, 150)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            self.lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['precision'])
            self.lstm.fit(self.X_train_lstm, self.y_train_lstm, epochs=50, batch_size=8, validation_split=0.2, verbose=0)
            self.lstm.save('lstm_model.h5')
        
        # Evaluate LSTM
        y_pred_prob = self.lstm.predict(self.X_test_lstm, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        self.evaluation_results['lstm'] = {
            'accuracy': accuracy_score(self.y_test_lstm, y_pred),
            'precision': precision_score(self.y_test_lstm, y_pred),
            'recall': recall_score(self.y_test_lstm, y_pred),
            'f1': f1_score(self.y_test_lstm, y_pred)
        }
        # Generate LSTM signal
        lstm_prob = self.lstm.predict(self.feature_matrix[self.features].iloc[6:], verbose=0)
        self.indicators[self.symbol]['lstm_signal'] = pd.Series(lstm_prob.flatten(), index=self.feature_matrix.index[6:])
        return self

    def build_meta_model(self):
        """Build and train the meta-model using LightGBM to combine signals."""
        print("Building meta-model...")
        signals = [
            'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
            'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
            'inverse_cramer', 'momentum_21', 'mean_reversion_20'
        ]
        # Generate additional signals
        self.indicators[self.symbol]['rsi_signal'] = np.where(self.indicators[self.symbol]['rsi_14'] < 30, 1, 
                                                              np.where(self.indicators[self.symbol]['rsi_14'] > 70, -1, 0))
        self.indicators[self.symbol]['macd_cross'] = np.where(self.indicators[self.symbol]['macd'] > self.indicators[self.symbol]['macd_signal'], 1, -1)
        self.indicators[self.symbol]['bb_signal'] = np.where(self.data[self.symbol]['Close'] < self.indicators[self.symbol]['bb_lower_20'], 1, 
                                                            np.where(self.data[self.symbol]['Close'] > self.indicators[self.symbol]['bb_upper_20'], -1, 0))
        self.indicators[self.symbol]['yield_curve_signal'] = np.where(self.macro_data['yield_curve_10y2y'] > 0, 1, -1)
        self.indicators[self.symbol]['atr_signal'] = np.where(self.indicators[self.symbol]['atr_14'] > self.indicators[self.symbol]['atr_14'].shift(1), 1, -1)
        self.indicators[self.symbol]['obv_signal'] = np.where(self.indicators[self.symbol]['obv'] > self.indicators[self.symbol]['obv_ma_20'], 1, -1)
        self.indicators[self.symbol]['vix_signal'] = np.where(self.data['^VIX']['Close'].pct_change() < -0.05, 1, 
                                                              np.where(self.data['^VIX']['Close'].pct_change() > 0.05, -1, 0))
        self.indicators[self.symbol]['supertrend_cont'] = (self.data[self.symbol]['Close'] - self.indicators[self.symbol]['sma_20']) / self.data[self.symbol]['Close']
        
        # Prepare signal data
        signal_df = self.indicators[self.symbol][signals].loc[self.feature_matrix.index[6:]].dropna()
        target = self.target.loc[signal_df.index]
        
        scaler = StandardScaler()
        signal_df_scaled = pd.DataFrame(scaler.fit_transform(signal_df), columns=signals, index=signal_df.index)
        
        # Split for meta-model
        train_size = int(0.8 * len(signal_df_scaled))
        X_train = signal_df_scaled.iloc[:train_size]
        y_train = target.iloc[:train_size]
        X_test = signal_df_scaled.iloc[train_size:]
        y_test = target.iloc[train_size:]
        
        # Train or load meta-model
        if os.path.exists('meta_model.pkl') and not hasattr(self, 'retrain'):
            with open('meta_model.pkl', 'rb') as f:
                self.meta_model = pickle.load(f)
        else:
            self.meta_model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, feature_fraction=0.3)  # Cap ensemble influence
            self.meta_model.fit(X_train, y_train)
            with open('meta_model.pkl', 'wb') as f:
                pickle.dump(self.meta_model, f)
        
        # Evaluate meta-model
        y_pred = self.meta_model.predict(X_test)
        self.evaluation_results['meta'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        self.meta_feature_importances = dict(zip(signals, self.meta_model.feature_importances_ / self.meta_model.feature_importances_.sum()))
        return self

    def predict_market(self):
        """Make a market prediction using the meta-model."""
        print("Predicting market...")
        signals = [
            'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
            'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
            'inverse_cramer', 'momentum_21', 'mean_reversion_20'
        ]
        latest = self.indicators[self.symbol][signals].iloc[-1:]
        scaler = StandardScaler()
        latest_scaled = scaler.fit_transform(latest)
        pred_prob = self.meta_model.predict_proba(latest_scaled)[:, 1][0]
        pred = "UP" if pred_prob > 0.5 else "DOWN"
        confidence = pred_prob if pred == "UP" else 1 - pred_prob
        self.final_prediction = pred
        self.confidence = confidence
        return pred, confidence

    def evaluate_and_compare(self):
        """Evaluate models and compare with previous iterations."""
        print("\nModel Evaluation Results:")
        for model_name, metrics in self.evaluation_results.items():
            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
        
        print(f"\nFinal Prediction: Market will go {self.final_prediction} (Confidence: {self.confidence:.2%})")
        print("\nTop Meta-Model Signals:")
        for i, (sig, imp) in enumerate(sorted(self.meta_feature_importances.items(), key=lambda x: x[1], reverse=True), 1):
            print(f"{i}. {sig}: {imp:.4f}")

        # Compare with previous results
        previous_results_file = 'previous_results.pkl'
        if os.path.exists(previous_results_file):
            with open(previous_results_file, 'rb') as f:
                previous_results = pickle.load(f)
            print("\nComparison with Previous Iteration:")
            for model_name, metrics in previous_results.items():
                print(f"{model_name.upper()}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")

        with open(previous_results_file, 'wb') as f:
            pickle.dump(self.evaluation_results, f)

    def run(self, retrain=False):
        """Run the full analysis pipeline."""
        if retrain:
            self.retrain = True
            for f in ['ensemble_model.pkl', 'lstm_model.h5', 'meta_model.pkl']:
                if os.path.exists(f):
                    os.remove(f)
        self.fetch_data()
        self.generate_indicators()
        self.prepare_features()
        self.build_ensemble()
        self.build_lstm()
        self.build_meta_model()
        self.predict_market()
        self.evaluate_and_compare()
        return self

if __name__ == "__main__":
    system = EnhancedFinancialAnalysisSystemV6()
    system.run(retrain=True)