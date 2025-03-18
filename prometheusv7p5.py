'''
Version 7.5:
small iteration

fix the lstm overfitting.
Last time, it was predicting up everytime.
Change by weighing downs more.

We also need to boost the meta model

We can do this by increasing n_estimators and feature fraction




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
    def __init__(self, symbol='SPY', lookback_years=10):  # Increased to 5 years
        self.symbol = symbol
        self.lookback_years = lookback_years
        self.start_date = (dt.datetime.now() - dt.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        self.end_date = dt.datetime.now().strftime('%Y-%m-%d')
        self.features = []
        self.data = {}
        self.indicators = {}
        self.finnhub_client = finnhub.Client(api_key="cv9orrpr01qpd9s8hhggcv9orrpr01qpd9s8hhh0")
        self.fred = Fred(api_key="3173a18ce8eb6fd417c310f8f887f3a5")
        self.evaluation_results = {}

    def fetch_data(self):
        print("Fetching data...")
        with ThreadPoolExecutor() as executor:
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
        
        print("Fetching FRED macroeconomic data...")
        macro_series = {
            'yield_curve_10y2y': ('T10Y2Y', lambda x: x / 100),
            'yield_curve_10y3m': ('T10Y3M', lambda x: x / 100),
            'fed_funds': ('FEDFUNDS', lambda x: x / 100),
            'cpi': ('CPIAUCSL', lambda x: x.pct_change(12)),
            'unemployment': ('UNRATE', lambda x: x / 100),
            'gdp': ('GDP', lambda x: x.pct_change(4).reindex(self.data[self.symbol].index, method='ffill')),
            'indpro': ('INDPRO', lambda x: x.reindex(self.data[self.symbol].index, method='ffill')),
            'consumer_confidence': ('UMCSENT', lambda x: x.reindex(self.data[self.symbol].index, method='ffill')),
            'treasury_10y': ('DGS10', lambda x: x / 100),
            'treasury_3m': ('DGS3MO', lambda x: x / 100),
            'real_yield': ('REAINTRATREARAT10Y', lambda x: x.reindex(self.data[self.symbol].index, method='ffill'))
        }
        self.macro_data = pd.DataFrame(index=self.data[self.symbol].index)
        for key, (series_id, transform) in macro_series.items():
            try:
                series = self.fred.get_series(series_id)
                self.macro_data[key] = transform(series)
            except Exception as e:
                print(f"Warning: Failed to fetch FRED series '{series_id}' - {e}. Setting to NaN.")
                self.macro_data[key] = pd.Series(index=self.data[self.symbol].index)
        self.macro_data.fillna(method='ffill', inplace=True)
        self.macro_data.fillna(method='bfill', inplace=True)
        
        self.data[self.symbol].fillna(method='ffill', inplace=True)
        self.data[self.symbol].fillna(method='bfill', inplace=True)
        return self

    def generate_indicators(self):
        print("Generating indicators...")
        df = self.data[self.symbol]
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # Use monthly close for the target
        monthly_close = close.resample('M').last()
        self.indicators[self.symbol] = pd.DataFrame(index=df.index)
        self.indicators[self.symbol]['target'] = (monthly_close.pct_change(1).shift(-1) > 0).reindex(df.index, method='ffill').fillna(False)
        
        # Technical Indicators
        tech_count = 0
        for window in [5, 10, 14, 20, 50, 100, 200]:
            self.indicators[self.symbol][f'sma_{window}'] = close.rolling(window).mean()
            self.indicators[self.symbol][f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
            self.indicators[self.symbol][f'rsi_{window}'] = ta.momentum.RSIIndicator(close, window).rsi()
            self.indicators[self.symbol][f'roc_{window}'] = ta.momentum.ROCIndicator(close, window).roc()
            tech_count += 4
        for window in [10, 20, 50]:
            self.indicators[self.symbol][f'bb_upper_{window}'] = ta.volatility.BollingerBands(close, window).bollinger_hband()
            self.indicators[self.symbol][f'bb_lower_{window}'] = ta.volatility.BollingerBands(close, window).bollinger_lband()
            self.indicators[self.symbol][f'bb_width_{window}'] = ta.volatility.BollingerBands(close, window).bollinger_wband()
            tech_count += 3
        for window in [10, 28]:
            self.indicators[self.symbol][f'atr_{window}'] = ta.volatility.AverageTrueRange(high, low, close, window).average_true_range()
            tech_count += 1

        # More single-window indicators
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
        tech_count += 17  # from the single indicators above

        print(f"Technical indicators generated: {tech_count}")
        
        # Macro Indicators
        for col in self.macro_data.columns:
            self.indicators[self.symbol][f'macro_{col}'] = self.macro_data[col]
        print(f"Macro indicators added: {len(self.macro_data.columns)}")
        
        # Volatility Indicators
        vix = self.data['^VIX']['Close'].squeeze()
        self.indicators[self.symbol]['vix'] = vix
        self.indicators[self.symbol]['vix_ma_20'] = vix.rolling(20).mean()
        self.indicators[self.symbol]['vix_pct_change'] = vix.pct_change()
        self.indicators[self.symbol]['real_vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        self.indicators[self.symbol]['real_vol_50'] = close.pct_change().rolling(50).std() * np.sqrt(252)
        vol_count = 5
        print(f"Volatility indicators generated: {vol_count}")
        
        # Quant-like Signals
        self.indicators[self.symbol]['momentum_21'] = close.pct_change(21)
        self.indicators[self.symbol]['momentum_63'] = close.pct_change(63)
        self.indicators[self.symbol]['mean_reversion_20'] = (close - self.indicators[self.symbol]['sma_20']) / self.indicators[self.symbol]['sma_20']
        self.indicators[self.symbol]['mean_reversion_50'] = (close - self.indicators[self.symbol]['sma_50']) / self.indicators[self.symbol]['sma_50']
        self.indicators[self.symbol]['breadth_signal'] = self.data['A-D']['Close'].rolling(20).mean().pct_change()
        quant_count = 5
        print(f"Quant-like signals generated: {quant_count}")
        
        # Sentiment Analysis
        news_data = self.finnhub_client.company_news(self.symbol, _from=self.start_date, to=self.end_date)
        if news_data:
            news_sent = pd.Series(
                [textblob.TextBlob(n['summary']).sentiment.polarity for n in news_data if 'summary' in n],
                index=[pd.to_datetime(n['datetime'], unit='s') for n in news_data if 'summary' in n]
            )
            # average sentiment by day
            news_sent = news_sent.groupby(news_sent.index.date).mean()
            news_sent = pd.Series(news_sent, index=pd.to_datetime(news_sent.index)).reindex(df.index, method='nearest').fillna(0)
        else:
            print(f"No news data for {self.symbol}. Setting news_sentiment to 0.")
            news_sent = pd.Series(0, index=df.index)
        self.indicators[self.symbol]['news_sentiment'] = news_sent
        self.indicators[self.symbol]['inverse_cramer'] = np.where(news_sent > 0.1, -1, np.where(news_sent < -0.1, 1, 0))
        sent_count = 2
        print(f"Sentiment indicators generated: {sent_count}")
        
        total_features = tech_count + len(self.macro_data.columns) + vol_count + quant_count + sent_count + 1  # +1 for target
        print(f"Total features (including target): {total_features}")
        
        # Final cleanup
        self.indicators[self.symbol].fillna(method='ffill', inplace=True)
        self.indicators[self.symbol].fillna(0, inplace=True)
        return self

    def create_sequences(self, X, y=None, timesteps=6):
        """
        Helper method to transform a dataframe of features X (and optional target y)
        into a 3D array of shape (samples, timesteps, features) suitable for LSTM.
        If y is provided, also return the matching target array.
        """
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X.iloc[i:(i + timesteps)].values)
            if y is not None:
                ys.append(y.iloc[i + timesteps])
        if y is not None:
            return np.array(Xs), np.array(ys)
        else:
            return np.array(Xs)

    def prepare_features(self):
        print("Preparing features...")
        # Drop initial 200 rows for stable indicators
        df = self.indicators[self.symbol].iloc[200:]
        self.features = [col for col in df.columns if col != 'target']
        print(f"Raw features available: {len(self.features)}")
        
        X = df[self.features]
        y = df['target']
        
        # Select top k features (but your code is setting k=min(150, len(self.features)))
        selector = SelectKBest(f_classif, k=min(150, len(self.features)))
        selector.fit(X, y)
        self.features = [self.features[i] for i in selector.get_support(indices=True)]
        print(f"Selected features: {len(self.features)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[self.features])
        self.feature_matrix = pd.DataFrame(X_scaled, index=df.index, columns=self.features)
        self.target = y
        
        # Resample monthly
        monthly_features = self.feature_matrix.resample('M').last()
        monthly_target = self.target.resample('M').last()
        
        n = len(monthly_features)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        # Ensemble training/validation/test sets
        self.X_train_ens = monthly_features.iloc[:train_end]
        self.X_val_ens = monthly_features.iloc[train_end:val_end]
        self.X_test_ens = monthly_features.iloc[val_end:]
        self.y_train_ens = monthly_target.iloc[:train_end]
        self.y_val_ens = monthly_target.iloc[train_end:val_end]
        self.y_test_ens = monthly_target.iloc[val_end:]
        
        # For the meta model we combine train+val
        self.X_train_meta = monthly_features.iloc[:val_end]
        self.X_test_meta = monthly_features.iloc[val_end:]
        self.y_train_meta = monthly_target.iloc[:val_end]
        self.y_test_meta = monthly_target.iloc[val_end:]
        
        # Create LSTM sequences
        self.X_train_lstm, self.y_train_lstm = self.create_sequences(self.X_train_meta, self.y_train_meta, timesteps=6)
        self.X_test_lstm, self.y_test_lstm = self.create_sequences(self.X_test_meta, self.y_test_meta, timesteps=6)
        
        print(f"X_train_lstm shape: {self.X_train_lstm.shape}, X_test_lstm shape: {self.X_test_lstm.shape}")
        return self

    def build_ensemble(self):
        print("Building ensemble...")
        if os.path.exists('ensemble_model.pkl') and not hasattr(self, 'retrain'):
            with open('ensemble_model.pkl', 'rb') as f:
                self.ensemble = pickle.load(f)
        else:
            self.ensemble = XGBClassifier(n_estimators=200, max_depth=5, reg_lambda=1,
                                          random_state=42, eval_metric='logloss')
            self.ensemble.fit(self.X_train_ens, self.y_train_ens,
                              eval_set=[(self.X_val_ens, self.y_val_ens)],
                              verbose=False)
            with open('ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble, f)
        
        y_pred = self.ensemble.predict(self.X_test_ens)
        print(f"Ensemble test size: {len(self.y_test_ens)}, Positives: {self.y_test_ens.sum()}")
        self.evaluation_results['ensemble'] = {
            'accuracy': accuracy_score(self.y_test_ens, y_pred),
            'precision': precision_score(self.y_test_ens, y_pred),
            'recall': recall_score(self.y_test_ens, y_pred),
            'f1': f1_score(self.y_test_ens, y_pred)
        }
        
        # Probability on full feature matrix (monthly)
        ensemble_prob = self.ensemble.predict_proba(self.feature_matrix[self.features])[:, 1]
        self.indicators[self.symbol]['ensemble_signal'] = pd.Series(ensemble_prob, index=self.feature_matrix.index)
        return self

    def build_lstm(self):
        print("Building LSTM...")
        n_features = self.feature_matrix.shape[1]
        print(f"LSTM input shape: (6, {n_features})")
        
        if os.path.exists('lstm_model.h5') and not hasattr(self, 'retrain'):
            self.lstm = tf.keras.models.load_model('lstm_model.h5')
        else:
            self.lstm = Sequential([
                LSTM(128, return_sequences=True, input_shape=(6, n_features)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            self.lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['precision'])
            self.lstm.fit(self.X_train_lstm, self.y_train_lstm, epochs=20, batch_size=8, 
              validation_split=0.2, class_weight={0: 2, 1: 1}, verbose=0)
            self.lstm.save('lstm_model.h5')
        
        if self.X_test_lstm.size > 0:
            y_pred_prob = self.lstm.predict(self.X_test_lstm, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int)

            print(f"LSTM y_test_lstm: {self.y_test_lstm}")
            print(f"LSTM y_pred: {y_pred.flatten()}")
            print(f"LSTM y_pred_prob: {y_pred_prob.flatten()}")
            print(f"LSTM test size: {len(self.y_test_lstm)}, Positives: {self.y_test_lstm.sum()}")

            self.evaluation_results['lstm'] = {
                'accuracy': accuracy_score(self.y_test_lstm, y_pred),
                'precision': precision_score(self.y_test_lstm, y_pred, zero_division=0),
                'recall': recall_score(self.y_test_lstm, y_pred, zero_division=0),
                'f1': f1_score(self.y_test_lstm, y_pred, zero_division=0)
            }
        else:
            print("Warning: X_test_lstm is empty. Skipping LSTM evaluation.")
            self.evaluation_results['lstm'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Predict LSTM signal for the entire feature_matrix
        X_full_lstm = self.create_sequences(self.feature_matrix, timesteps=6)
        lstm_prob = self.lstm.predict(X_full_lstm, verbose=0)
        # Align the resulting probabilities with the correct dates (shift by 6)
        self.indicators[self.symbol]['lstm_signal'] = pd.Series(
            lstm_prob.flatten(), index=self.feature_matrix.index[6:]
        )
        return self

    def build_meta_model(self):
        print("Building meta-model...")
        # Prepare the signals used in meta-model
        signals = [
            'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
            'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
            'inverse_cramer', 'momentum_21', 'mean_reversion_20'
        ]
        
        # Generate simpler signals for RSI, MACD cross, Bollinger, yield_curve, etc.
        self.indicators[self.symbol]['rsi_signal'] = np.where(
            self.indicators[self.symbol]['rsi_14'] < 30, 1,
            np.where(self.indicators[self.symbol]['rsi_14'] > 70, -1, 0)
        )
        self.indicators[self.symbol]['macd_cross'] = np.where(
            self.indicators[self.symbol]['macd'] > self.indicators[self.symbol]['macd_signal'], 1, -1
        )

        # Ensure close_series is a Series aligned to the indicators index
        close_series = self.data[self.symbol]['Close'].squeeze().reindex(self.indicators[self.symbol].index)
        # Bollinger band signal
        self.indicators[self.symbol]['bb_signal'] = np.where(
            close_series < self.indicators[self.symbol]['bb_lower_20'], 1,
            np.where(close_series > self.indicators[self.symbol]['bb_upper_20'], -1, 0)
        )

        # Yield curve signal
        self.indicators[self.symbol]['yield_curve_signal'] = pd.Series(
            np.where(self.macro_data['yield_curve_10y2y'] > 0, 1, -1),
            index=self.macro_data.index
        ).reindex(self.indicators[self.symbol].index, method='ffill').fillna(0)

        # ATR signal
        self.indicators[self.symbol]['atr_signal'] = np.where(
            self.indicators[self.symbol]['atr_14'] > self.indicators[self.symbol]['atr_14'].shift(1), 1, -1
        )

        # OBV signal
        self.indicators[self.symbol]['obv_signal'] = np.where(
            self.indicators[self.symbol]['obv'] > self.indicators[self.symbol]['obv_ma_20'], 1, -1
        )

        # VIX signal
        vix_series = self.data['^VIX']['Close'].squeeze().reindex(self.indicators[self.symbol].index)
        self.indicators[self.symbol]['vix_signal'] = np.where(
            vix_series.pct_change() < -0.05, 1,
            np.where(vix_series.pct_change() > 0.05, -1, 0)
        )

        # Supertrend-like continuity
        self.indicators[self.symbol]['supertrend_cont'] = (
            (close_series - self.indicators[self.symbol]['sma_20']) / close_series
        )

        # Now gather the final signals into a DataFrame aligned with feature_matrix (post-6 for LSTM, etc.)
        # We'll start the meta data from the same index as the ensemble & LSTM signals appear:
        signal_df = self.indicators[self.symbol][signals].resample('M').last().dropna()
       
        
        valid_index = self.feature_matrix.index.intersection(signal_df.index)
        signal_df = signal_df.loc[valid_index]
        target = self.target.resample('M').last().loc[signal_df.index]

        # Scale these signals
        scaler = StandardScaler()
        signal_df_scaled = pd.DataFrame(
            scaler.fit_transform(signal_df),
            columns=signals,
            index=signal_df.index
        )

        # Train-test split on these signals
        train_size = int(0.8 * len(signal_df_scaled))
        X_train = signal_df_scaled.iloc[:train_size]
        y_train = target.iloc[:train_size]
        X_test = signal_df_scaled.iloc[train_size:]
        y_test = target.iloc[train_size:]
        
        print(f"Meta test size: {len(y_test)}, Positives: {y_test.sum()}")
        
        # Build or load meta-model
        if os.path.exists('meta_model.pkl') and not hasattr(self, 'retrain'):
            with open('meta_model.pkl', 'rb') as f:
                self.meta_model = pickle.load(f)
        else:
            self.meta_model = LGBMClassifier(n_estimators=200, max_depth=5, random_state=42, feature_fraction=0.5)
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

        # Store feature importances
        self.meta_feature_importances = dict(
            zip(signals, self.meta_model.feature_importances_ / self.meta_model.feature_importances_.sum())
        )
        return self

    def predict_market(self):
        print("Predicting market...")
        # These signals must match exactly what was used in the meta-model:
        signals = [
            'ensemble_signal', 'lstm_signal', 'rsi_signal', 'macd_cross', 'bb_signal',
            'yield_curve_signal', 'atr_signal', 'obv_signal', 'vix_signal', 'supertrend_cont',
            'inverse_cramer', 'momentum_21', 'mean_reversion_20'
        ]
        
        latest = self.indicators[self.symbol][signals].iloc[-1:].copy()
        # We must scale the same way we scaled in build_meta_model, but quick approach is a fresh StandardScaler:
        scaler = StandardScaler()
        latest_scaled = scaler.fit_transform(latest)
        
        pred_prob = self.meta_model.predict_proba(latest_scaled)[:, 1][0]
        pred = "UP" if pred_prob > 0.5 else "DOWN"
        confidence = pred_prob if pred == "UP" else 1 - pred_prob
        self.final_prediction = pred
        self.confidence = confidence
        return pred, confidence

    def evaluate_and_compare(self):
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
