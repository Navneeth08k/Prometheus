'''
Version 5:

Uses xgboost for ensemble and random forest for meta model.
'''

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import warnings
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import ta
import textblob
from concurrent.futures import ThreadPoolExecutor
import finnhub
from fredapi import Fred
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

class EnhancedFinancialAnalysisSystem:
    def __init__(self, symbols=['SPY', 'QQQ', 'IWM', 'DIA'], lookback_years=3):
        """Initialize the system with symbols and lookback period."""
        self.symbols = symbols
        self.lookback_years = lookback_years
        self.start_date = (dt.datetime.now() - dt.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        self.end_date = dt.datetime.now().strftime('%Y-%m-%d')
        self.features = []
        self.data = {}
        self.indicators = {}
        self.finnhub_client = finnhub.Client(api_key="cv9orrpr01qpd9s8hhggcv9orrpr01qpd9s8hhh0")  # Replace with your Finnhub API key
        self.fred = Fred(api_key="3173a18ce8eb6fd417c310f8f887f3a5")  # Replace with your FRED API key
        self.evaluation_results = {}

    def fetch_data(self):
        """Fetch historical price and macroeconomic data."""
        print("Fetching data...")
        with ThreadPoolExecutor() as executor:
            self.data = {symbol: yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False) 
                         for symbol in self.symbols}
            self.data['^VIX'] = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
            try:
                self.data['A-D'] = yf.download('^NYAD', start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)
                if self.data['A-D'].empty:
                    print("Warning: '^NYAD' data is empty. Breadth signal will be set to 0.")
                    self.data['A-D'] = pd.DataFrame(index=self.data[self.symbols[0]].index, columns=['Close']).fillna(0)
            except Exception as e:
                print(f"Warning: Failed to fetch '^NYAD' - {e}. Breadth signal will be set to 0.")
                self.data['A-D'] = pd.DataFrame(index=self.data[self.symbols[0]].index, columns=['Close']).fillna(0)
        
        print("Fetching FRED macroeconomic data...")
        self.macro_data = pd.DataFrame({
            'yield_curve_10y2y': self.fred.get_series('T10Y2Y') / 100,
            'fed_funds': self.fred.get_series('FEDFUNDS') / 100,
            'cpi': self.fred.get_series('CPIAUCSL').pct_change(12),
            'unemployment': self.fred.get_series('UNRATE') / 100
        }).reindex(self.data[self.symbols[0]].index, method='ffill').fillna(method='bfill')
        
        for symbol in self.symbols[:]:
            if self.data[symbol].empty:
                print(f"No data for {symbol}. Removing from analysis.")
                self.symbols.remove(symbol)
            else:
                self.data[symbol].fillna(method='ffill', inplace=True)
                self.data[symbol].fillna(method='bfill', inplace=True)
        return self

    def generate_indicators(self):
        """Generate technical, macro, and sentiment indicators."""
        print("Generating indicators...")
        for symbol in self.symbols:
            df = self.data[symbol]
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            
            self.indicators[symbol] = pd.DataFrame(index=df.index)
            self.indicators[symbol]['return_1d'] = close.pct_change()
            self.indicators[symbol]['target'] = self.indicators[symbol]['return_1d'].shift(-1) > 0
            
            # Comprehensive Technical Indicators
            for window in [5, 10, 14, 20, 50, 200]:
                self.indicators[symbol][f'ma_{window}'] = close.rolling(window).mean()
                self.indicators[symbol][f'rsi_{window}'] = ta.momentum.RSIIndicator(close, window).rsi()
            self.indicators[symbol]['macd'] = ta.trend.MACD(close).macd()
            self.indicators[symbol]['macd_signal'] = ta.trend.MACD(close).macd_signal()
            self.indicators[symbol]['bb_upper_20'] = ta.volatility.BollingerBands(close, 20).bollinger_hband()
            self.indicators[symbol]['bb_lower_20'] = ta.volatility.BollingerBands(close, 20).bollinger_lband()
            self.indicators[symbol]['atr_14'] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
            self.indicators[symbol]['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            self.indicators[symbol]['obv_ma_20'] = self.indicators[symbol]['obv'].rolling(20).mean()
            
            # Macro Indicators
            for col in self.macro_data.columns:
                self.indicators[symbol][f'macro_{col}'] = self.macro_data[col]
            
            # Sentiment Analysis
            news_data = self.finnhub_client.company_news(symbol, _from=self.start_date, to=self.end_date)
            if news_data:
                news_sent = pd.Series(
                    [textblob.TextBlob(n['summary']).sentiment.polarity for n in news_data if 'summary' in n],
                    index=[pd.to_datetime(n['datetime'], unit='s') for n in news_data if 'summary' in n]
                )
                news_sent = news_sent.groupby(news_sent.index.date).mean()
                news_sent = pd.Series(news_sent, index=pd.to_datetime(news_sent.index)).reindex(df.index, method='nearest').fillna(0)
            else:
                print(f"No news data for {symbol}. Setting news_sentiment to 0.")
                news_sent = pd.Series(0, index=df.index)
            self.indicators[symbol]['news_sentiment'] = news_sent
            
            # Meta-Model Signals (for primary symbol only)
            if symbol == self.symbols[0]:
                # Supertrend
                atr = self.indicators[symbol]['atr_14']
                basic_upper = (high + low) / 2 + 3 * atr
                basic_lower = (high + low) / 2 - 3 * atr
                supertrend = pd.Series(index=close.index, dtype=float)
                direction = pd.Series(index=close.index, dtype=int)
                supertrend.iloc[0] = basic_upper.iloc[0]
                direction.iloc[0] = -1
                for i in range(1, len(close)):
                    if direction.iloc[i-1] == 1 and close.iloc[i-1] > supertrend.iloc[i-1]:
                        supertrend.iloc[i] = basic_lower.iloc[i]
                        direction.iloc[i] = 1
                    elif direction.iloc[i-1] == -1 and close.iloc[i-1] < supertrend.iloc[i-1]:
                        supertrend.iloc[i] = basic_upper.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                        direction.iloc[i] = direction.iloc[i-1]
                        if direction.iloc[i] == 1 and supertrend.iloc[i] > basic_lower.iloc[i]:
                            supertrend.iloc[i] = basic_lower.iloc[i]
                        elif direction.iloc[i] == -1 and supertrend.iloc[i] < basic_upper.iloc[i]:
                            supertrend.iloc[i] = basic_upper.iloc[i]
                self.indicators[symbol]['supertrend_signal'] = direction
                self.indicators[symbol]['rsi_signal'] = np.where(self.indicators[symbol]['rsi_14'] < 30, 1, 
                                                                np.where(self.indicators[symbol]['rsi_14'] > 70, -1, 0))
                self.indicators[symbol]['macd_cross'] = np.where(self.indicators[symbol]['macd'] > self.indicators[symbol]['macd_signal'], 1, -1)
                self.indicators[symbol]['bb_signal'] = np.where(close < self.indicators[symbol]['bb_lower_20'], 1, 
                                                               np.where(close > self.indicators[symbol]['bb_upper_20'], -1, 0))
                self.indicators[symbol]['yield_curve_signal'] = np.where(self.macro_data['yield_curve_10y2y'] > 0, 1, -1)
                self.indicators[symbol]['atr_signal'] = np.where(self.indicators[symbol]['atr_14'] > self.indicators[symbol]['atr_14'].shift(1), 1, -1)
                self.indicators[symbol]['obv_signal'] = np.where(self.indicators[symbol]['obv'] > self.indicators[symbol]['obv_ma_20'], 1, -1)
                
                # Additional Signals
                vix = self.data['^VIX']['Close'].squeeze()
                self.indicators[symbol]['vix_signal'] = np.where(vix.pct_change() < -0.05, 1, np.where(vix.pct_change() > 0.05, -1, 0))
                self.indicators[symbol]['supertrend_cont'] = (close - supertrend) / close
                ad_line = self.data['A-D']['Close'].squeeze()
                self.indicators[symbol]['breadth_signal'] = ad_line.rolling(20).mean().pct_change() > 0
                self.indicators[symbol]['sentiment_delta'] = self.indicators[symbol]['news_sentiment'].diff()
                vol_ma = volume.rolling(20).mean()
                self.indicators[symbol]['vol_spike'] = np.where(volume > 2 * vol_ma, 1, 0)
            
            self.indicators[symbol].fillna(method='ffill', inplace=True)
            self.indicators[symbol].fillna(0, inplace=True)
        return self

    def prepare_features(self):
        """Prepare feature matrix for modeling."""
        print("Preparing features...")
        all_features = pd.DataFrame()
        for symbol in self.symbols:
            indicators = self.indicators[symbol].copy()
            indicators.columns = [f'{symbol}_{col}' if col != 'target' else col for col in indicators.columns]
            if all_features.empty:
                all_features = indicators
            else:
                all_features = all_features.join(indicators.drop('target', axis=1, errors='ignore'))
        
        all_features = all_features.iloc[200:]  # Skip initial rows with NaNs from indicators
        self.features = [col for col in all_features.columns if col != 'target']
        
        X = all_features[self.features]
        y = all_features['target']
        selector = SelectKBest(f_classif, k=50)
        selector.fit(X, y)
        self.features = [self.features[i] for i in selector.get_support(indices=True)]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[self.features])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        self.feature_matrix = pd.DataFrame(X_scaled, index=all_features.index, columns=self.features)
        self.target = y
        return self

    def build_ensemble(self):
        """Build and train an ensemble model."""
        print("Building ensemble...")
        if os.path.exists('ensemble_model.pkl') and not hasattr(self, 'retrain'):
            with open('ensemble_model.pkl', 'rb') as f:
                self.ensemble = pickle.load(f)
        else:
            self.ensemble = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss')
            self.ensemble.fit(self.X_train, self.y_train)
            with open('ensemble_model.pkl', 'wb') as f:
                pickle.dump(self.ensemble, f)
        
        train_prob = self.ensemble.predict_proba(self.X_train)[:, 1]
        test_prob = self.ensemble.predict_proba(self.X_test)[:, 1]
        ensemble_prob = np.concatenate([train_prob, test_prob])
        self.indicators[self.symbols[0]]['ensemble_signal'] = pd.Series(ensemble_prob, index=self.feature_matrix.index)
        
        # Evaluate ensemble
        y_pred = self.ensemble.predict(self.X_test)
        self.evaluation_results['ensemble'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }
        return self

    def build_meta_model(self):
        """Build and train a meta-model using diverse signals."""
        print("Building meta-model...")
        symbol = self.symbols[0]
        signals = [
            'ensemble_signal', 'supertrend_signal', 'rsi_signal', 'macd_cross', 
            'bb_signal', 'yield_curve_signal', 'atr_signal', 'obv_signal', 
            'vix_signal', 'supertrend_cont', 'breadth_signal', 'sentiment_delta', 'vol_spike'
        ]
        signal_df = self.indicators[symbol][signals].loc[self.feature_matrix.index].dropna()
        target = self.target.loc[signal_df.index]
        
        scaler = StandardScaler()
        signal_df_scaled = pd.DataFrame(scaler.fit_transform(signal_df), columns=signals, index=signal_df.index)
        
        train_size = int(0.8 * len(signal_df_scaled))
        X_train = signal_df_scaled.iloc[:train_size]
        y_train = target.iloc[:train_size]
        X_test = signal_df_scaled.iloc[train_size:]
        y_test = target.iloc[train_size:]
        
        if os.path.exists('meta_model.pkl') and not hasattr(self, 'retrain'):
            with open('meta_model.pkl', 'rb') as f:
                self.meta_model = pickle.load(f)
        else:
            self.meta_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self.meta_model.fit(X_train, y_train)
            with open('meta_model.pkl', 'wb') as f:
                pickle.dump(self.meta_model, f)
        
        meta_pred = self.meta_model.predict(X_test)
        self.evaluation_results['meta'] = {
            'accuracy': accuracy_score(y_test, meta_pred),
            'precision': precision_score(y_test, meta_pred),
            'recall': recall_score(y_test, meta_pred),
            'f1': f1_score(y_test, meta_pred)
        }
        self.meta_feature_importances = dict(zip(signals, self.meta_model.feature_importances_))
        return self

    def predict_market(self):
        """Make a market prediction using the meta-model."""
        print("Predicting market...")
        symbol = self.symbols[0]
        signals = [
            'ensemble_signal', 'supertrend_signal', 'rsi_signal', 'macd_cross', 
            'bb_signal', 'yield_curve_signal', 'atr_signal', 'obv_signal', 
            'vix_signal', 'supertrend_cont', 'breadth_signal', 'sentiment_delta', 'vol_spike'
        ]
        latest = self.indicators[symbol][signals].iloc[-1:]
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

        # Save current results
        with open(previous_results_file, 'wb') as f:
            pickle.dump(self.evaluation_results, f)

    def run(self, retrain=False):
        """Run the full analysis pipeline."""
        if retrain:
            self.retrain = True
            for f in ['ensemble_model.pkl', 'meta_model.pkl']:
                if os.path.exists(f):
                    os.remove(f)
        self.fetch_data()
        self.generate_indicators()
        self.prepare_features()
        self.build_ensemble()
        self.build_meta_model()
        self.predict_market()
        self.evaluate_and_compare()
        return self

if __name__ == "__main__":
    system = EnhancedFinancialAnalysisSystem()
    system.run(retrain=True)