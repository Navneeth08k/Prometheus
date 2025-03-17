import pandas as pd, numpy as np, yfinance as yf, requests, json, datetime as dt, warnings, os, pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import ta
import textblob
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings; warnings.filterwarnings('ignore')

# Compact financial analysis system
class CompactFinancialAnalysisSystem:
    def __init__(self, symbols=['SPY', 'QQQ', 'DIA'], lookback_years=3, macro_indicators=True, sentiment_analysis=True, feature_selection=True, n_features=50):
        self.symbols = symbols
        self.lookback_years = lookback_years
        self.macro_indicators = macro_indicators
        self.sentiment_analysis = sentiment_analysis
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.start_date = (dt.datetime.now() - dt.timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
        self.end_date = dt.datetime.now().strftime('%Y-%m-%d')
        self.features = []
        self.models = {}
        self.data = {}
        self.indicators = {}
        self.feature_importances = {}
        
    def fetch_data(self):
        # Fetch market data for all symbols in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            self.data = {symbol: yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False) for symbol in self.symbols}
        
        # Fetch macro data
        if self.macro_indicators:
            try:
                # FRED API (replace with your API key if you have one)
                fred_api_key = os.environ.get('FRED_API_KEY', '')
                self.macro_data = self._fetch_macro_data(fred_api_key)
            except:
                print("Warning: Unable to fetch macro data, proceeding without it")
                self.macro_indicators = False
        
        # Check for missing data and handle it
        for symbol in self.symbols:
            if self.data[symbol].empty:
                print(f"No data found for {symbol}. Removing from analysis.")
                self.symbols.remove(symbol)
            else:
                self.data[symbol].fillna(method='ffill', inplace=True)
                self.data[symbol].fillna(method='bfill', inplace=True)
        
        return self
    
    def _fetch_macro_data(self, api_key):
        # Simulated macro data - in a real implementation, you would use FRED API or similar
        # Create synthetic macro data aligned with market data dates
        macro_data = pd.DataFrame(index=self.data[self.symbols[0]].index)
        
        # Generate synthetic macro indicators
        macro_data['gdp_growth'] = np.random.normal(2.5, 1.5, size=len(macro_data)) / 100  # Quarterly GDP growth
        macro_data['inflation'] = np.random.normal(2.0, 0.8, size=len(macro_data)) / 100   # Monthly inflation
        macro_data['unemployment'] = np.random.normal(5.0, 1.0, size=len(macro_data)) / 100 # Monthly unemployment
        macro_data['interest_rate'] = np.random.normal(2.0, 1.2, size=len(macro_data)) / 100 # Fed funds rate
        macro_data['bond_10y_yield'] = np.random.normal(3.0, 0.5, size=len(macro_data)) / 100 # 10-year Treasury yield
        macro_data['pmi'] = np.random.normal(52, 3, size=len(macro_data))  # PMI
        
        # Add trend components and seasonality to make it more realistic
        for col in macro_data.columns:
            # Add trend
            macro_data[col] = macro_data[col] + np.linspace(0, 0.02, len(macro_data)) if col != 'pmi' else macro_data[col] + np.linspace(0, 2, len(macro_data))
            # Add seasonality
            days = np.arange(len(macro_data))
            seasonal_component = 0.005 * np.sin(2 * np.pi * days / 365) if col != 'pmi' else 0.5 * np.sin(2 * np.pi * days / 365)
            macro_data[col] = macro_data[col] + seasonal_component
        
        return macro_data
    
    def generate_indicators(self):
        for symbol in self.symbols:
            df = self.data[symbol].copy()
            
            # Ensure single-column selections are Series
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
            volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
            high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
            low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
            
            # Initialize indicators dictionary for this symbol
            self.indicators[symbol] = pd.DataFrame(index=df.index)
            
            # Price and volume features (basic)
            self.indicators[symbol]['return_1d'] = close.pct_change(1)
            self.indicators[symbol]['return_5d'] = close.pct_change(5)
            self.indicators[symbol]['return_10d'] = close.pct_change(10)
            self.indicators[symbol]['return_20d'] = close.pct_change(20)
            self.indicators[symbol]['return_60d'] = close.pct_change(60)
            self.indicators[symbol]['vol_change_1d'] = volume.pct_change(1)
            self.indicators[symbol]['vol_change_5d'] = volume.pct_change(5)
            self.indicators[symbol]['price_vol_corr_20d'] = close.rolling(20).corr(volume)
            
            # Calculate target variable (next day return > 0)
            self.indicators[symbol]['target'] = self.indicators[symbol]['return_1d'].shift(-1) > 0
            
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                self.indicators[symbol][f'ma_{window}'] = close.rolling(window).mean()
                self.indicators[symbol][f'ma_ratio_{window}'] = close / self.indicators[symbol][f'ma_{window}']
                self.indicators[symbol][f'vol_ma_{window}'] = volume.rolling(window).mean()
                self.indicators[symbol][f'vol_ratio_{window}'] = volume / self.indicators[symbol][f'vol_ma_{window}']
            
            # Exponential moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                self.indicators[symbol][f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
                self.indicators[symbol][f'ema_ratio_{window}'] = close / self.indicators[symbol][f'ema_{window}']
                self.indicators[symbol][f'vol_ema_{window}'] = volume.ewm(span=window, adjust=False).mean()
            
            # MACD
            self.indicators[symbol]['macd'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            self.indicators[symbol]['macd_signal'] = self.indicators[symbol]['macd'].ewm(span=9, adjust=False).mean()
            self.indicators[symbol]['macd_hist'] = self.indicators[symbol]['macd'] - self.indicators[symbol]['macd_signal']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            for window in [7, 14, 21]:
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss
                self.indicators[symbol][f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            for window in [10, 20, 50]:
                self.indicators[symbol][f'bb_mid_{window}'] = close.rolling(window).mean()
                self.indicators[symbol][f'bb_std_{window}'] = close.rolling(window).std()
                self.indicators[symbol][f'bb_upper_{window}'] = self.indicators[symbol][f'bb_mid_{window}'] + 2 * self.indicators[symbol][f'bb_std_{window}']
                self.indicators[symbol][f'bb_lower_{window}'] = self.indicators[symbol][f'bb_mid_{window}'] - 2 * self.indicators[symbol][f'bb_std_{window}']
                self.indicators[symbol][f'bb_width_{window}'] = (self.indicators[symbol][f'bb_upper_{window}'] - self.indicators[symbol][f'bb_lower_{window}']) / self.indicators[symbol][f'bb_mid_{window}']
                self.indicators[symbol][f'bb_pct_{window}'] = (close - self.indicators[symbol][f'bb_lower_{window}']) / (self.indicators[symbol][f'bb_upper_{window}'] - self.indicators[symbol][f'bb_lower_{window}'])
            
            # ATR (Average True Range)
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            for window in [7, 14, 21]:
                self.indicators[symbol][f'atr_{window}'] = true_range.rolling(window).mean()
                self.indicators[symbol][f'atr_pct_{window}'] = self.indicators[symbol][f'atr_{window}'] / close
            
            # ADX (Average Directional Index) - simplified calculation
            for window in [7, 14, 21]:
                up_move = high.diff()
                down_move = low.diff(-1).abs()
                pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                tr = true_range
                pos_di = 100 * pd.Series(pos_dm).rolling(window).mean() / tr.rolling(window).mean()
                neg_di = 100 * pd.Series(neg_dm).rolling(window).mean() / tr.rolling(window).mean()
                dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
                self.indicators[symbol][f'adx_{window}'] = dx.rolling(window).mean()
            
            # VWAP (Volume Weighted Average Price) - daily
            df['vwap_daily'] = ((close + high + low) / 3 * volume).cumsum() / volume.cumsum()
            self.indicators[symbol]['vwap_ratio'] = close / df['vwap_daily']
            
            # OBV (On-Balance Volume)
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            self.indicators[symbol]['obv'] = obv
            for window in [10, 20, 50]:
                self.indicators[symbol][f'obv_ma_{window}'] = obv.rolling(window).mean()
                self.indicators[symbol][f'obv_ratio_{window}'] = obv / self.indicators[symbol][f'obv_ma_{window}']
            
            # Ichimoku Cloud components (simplified)
            high_9 = high.rolling(9).max()
            low_9 = low.rolling(9).min()
            self.indicators[symbol]['ichimoku_conversion'] = (high_9 + low_9) / 2
            high_26 = high.rolling(26).max()
            low_26 = low.rolling(26).min()
            self.indicators[symbol]['ichimoku_base'] = (high_26 + low_26) / 2
            self.indicators[symbol]['ichimoku_span_a'] = ((self.indicators[symbol]['ichimoku_conversion'] + self.indicators[symbol]['ichimoku_base']) / 2).shift(26)
            self.indicators[symbol]['ichimoku_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            
            # Add more indicators using the ta library
            self.indicators[symbol]['cci_20'] = ta.trend.cci(high, low, close, window=20)
            self.indicators[symbol]['roc_20'] = ta.momentum.roc(close, window=20)
            self.indicators[symbol]['stoch_k'] = ta.momentum.stoch(high, low, close)
            self.indicators[symbol]['stoch_d'] = ta.momentum.stoch_signal(high, low, close)
            self.indicators[symbol]['williamsr_14'] = ta.momentum.williams_r(high, low, close, lbp=14)
            
            # Trend
            # Trend
            self.indicators[symbol]['aroon_up'] = ta.trend.aroon_up(high=high, low=low, window=25)
            self.indicators[symbol]['aroon_down'] = ta.trend.aroon_down(high=high, low=low, window=25)
            self.indicators[symbol]['dpo_20'] = ta.trend.dpo(close, window=20)
            self.indicators[symbol]['kst'] = ta.trend.kst(close)
            self.indicators[symbol]['kst_sig'] = ta.trend.kst_sig(close)
            
            # Volatility
            self.indicators[symbol]['ulcer_14'] = ta.volatility.ulcer_index(close, window=14)
            
            # Volume
            self.indicators[symbol]['ease_of_movement'] = ta.volume.ease_of_movement(high, low, volume, window=14)
            self.indicators[symbol]['force_index_13'] = ta.volume.force_index(close, volume, window=13)
            self.indicators[symbol]['mfi_14'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
            self.indicators[symbol]['nvi'] = ta.volume.negative_volume_index(close, volume)
            
            # Statistical features
            for window in [10, 20, 50]:
                self.indicators[symbol][f'return_skew_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).skew()
                self.indicators[symbol][f'return_kurt_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).kurt()
                self.indicators[symbol][f'return_vol_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).std()
                self.indicators[symbol][f'return_z_{window}'] = (self.indicators[symbol]['return_1d'] - self.indicators[symbol]['return_1d'].rolling(window).mean()) / self.indicators[symbol]['return_1d'].rolling(window).std()
                mean_return = self.indicators[symbol]['return_1d'].rolling(window).mean() * 252  # Annualized
                std_return = self.indicators[symbol]['return_1d'].rolling(window).std() * np.sqrt(252)  # Annualized
                self.indicators[symbol][f'sharpe_{window}'] = mean_return / std_return
            
            # Fundamental indicators (simulated)
            fundamental_columns = ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_to_equity', 'free_cash_flow', 'eps_surprise', 'dividend_yield']
            for col in fundamental_columns:
                base_value = np.random.normal(15, 5) if col == 'pe_ratio' else np.random.normal(0.1, 0.05)
                trend = np.linspace(-0.02, 0.02, len(df))
                noise = np.random.normal(0, 0.01, len(df))
                self.indicators[symbol][col] = base_value + trend + noise
            
            # Add macro indicators if available
            if self.macro_indicators and hasattr(self, 'macro_data'):
                aligned_macro_data = self.macro_data.reindex(self.indicators[symbol].index).interpolate(method='linear')
                for col in aligned_macro_data.columns:
                    self.indicators[symbol][f'macro_{col}'] = aligned_macro_data[col]
                    self.indicators[symbol][f'macro_{col}_roc_20'] = aligned_macro_data[col].pct_change(20)
                    self.indicators[symbol][f'macro_{col}_roc_60'] = aligned_macro_data[col].pct_change(60)
            
            # Add sentiment analysis (simulated)
            if self.sentiment_analysis:
                self.indicators[symbol]['news_sentiment'] = np.random.normal(0.1, 0.3, len(df))
                self.indicators[symbol]['social_sentiment'] = np.random.normal(0.05, 0.4, len(df))
                self.indicators[symbol]['analyst_sentiment'] = np.random.normal(0.2, 0.25, len(df))
                days = np.arange(len(df))
                lagged_returns = self.indicators[symbol]['return_20d'].shift(5).fillna(0)
                sentiment_trend = 0.4 * lagged_returns + 0.1 * np.sin(2 * np.pi * days / 180)
                self.indicators[symbol]['news_sentiment'] += sentiment_trend
                self.indicators[symbol]['social_sentiment'] += sentiment_trend + 0.05 * np.cos(2 * np.pi * days / 90)
                self.indicators[symbol]['analyst_sentiment'] += sentiment_trend - 0.03 * np.sin(2 * np.pi * days / 60)
                for col in ['news_sentiment', 'social_sentiment', 'analyst_sentiment']:
                    self.indicators[symbol][col] = np.clip(self.indicators[symbol][col], -1, 1)
                    for window in [5, 10, 20]:
                        self.indicators[symbol][f'{col}_ma_{window}'] = self.indicators[symbol][col].rolling(window).mean()
                self.indicators[symbol]['combined_sentiment'] = (self.indicators[symbol]['news_sentiment'] + 
                                                                self.indicators[symbol]['social_sentiment'] + 
                                                                self.indicators[symbol]['analyst_sentiment']) / 3
            
            # Clean up indicators
            self.indicators[symbol] = self.indicators[symbol].replace([np.inf, -np.inf], np.nan)
            self.indicators[symbol].fillna(method='ffill', inplace=True)
            self.indicators[symbol].fillna(method='bfill', inplace=True)
            self.indicators[symbol].fillna(0, inplace=True)
        
        return self
    
    def prepare_features(self):
        # Combine features from all symbols
        all_features = pd.DataFrame()
        
        for symbol in self.symbols:
            # Get indicators for this symbol
            indicators = self.indicators[symbol].copy()
            
            # Add symbol prefix to avoid column name conflicts
            indicators.columns = [f'{symbol}_{col}' if col != 'target' else col for col in indicators.columns]
            
            # If this is the first symbol, use it as the base
            if all_features.empty:
                all_features = indicators
            else:
                # Otherwise, merge with existing features, keeping the target from the first symbol
                indicators = indicators.drop('target', axis=1, errors='ignore')
                all_features = all_features.join(indicators)
        
        # Remove the first 200 rows to allow for indicator calculation
        all_features = all_features.iloc[200:].copy()
        
        # Store list of feature names (excluding target)
        self.features = [col for col in all_features.columns if col != 'target']
        
        # Feature selection if enabled
        if self.feature_selection and len(self.features) > self.n_features:
            X = all_features[self.features]
            y = all_features['target']
            
            # Use SelectKBest for feature selection
            selector = SelectKBest(f_classif, k=self.n_features)
            selector.fit(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            self.features = [self.features[i] for i in selected_indices]
            
            # Store feature importances
            self.feature_importances = {self.features[i]: selector.scores_[selected_indices[i]] for i in range(len(self.features))}
        
        # Split data
        X = all_features[self.features]
        y = all_features['target']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for training and prediction
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        
        # Store feature matrix and target
        self.feature_matrix = pd.DataFrame(X_scaled, index=all_features.index, columns=self.features)
        self.target = all_features['target']
        
        return self
    
    def build_models(self):
        # Base models for ensemble
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        
        # Create ensemble model
        self.models['ensemble'] = VotingClassifier(estimators=base_models, voting='soft')
        
        # Train ensemble model
        self.models['ensemble'].fit(self.X_train, self.y_train)
        
        # Store individual models for later use
        for name, model in base_models:
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        
        # Calculate and store feature importances from Random Forest
        if 'rf' in self.models:
            self.feature_importances = {feature: importance for feature, importance in zip(self.features, self.models['rf'].feature_importances_)}
        
        return self
    
    def evaluate_models(self):
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Compute metrics
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
        
        self.evaluation_results = results
        return self
    
    def predict_market(self):
        # Make prediction using the latest data
        latest_data = self.feature_matrix.iloc[-1:].values
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(latest_data)[0]
            prob = model.predict_proba(latest_data)[0][1] if hasattr(model, 'predict_proba') else 0.5
            predictions[name] = (pred, prob)
        
        # Ensemble prediction
        ensemble_pred = self.models['ensemble'].predict(latest_data)[0]
        ensemble_prob = self.models['ensemble'].predict_proba(latest_data)[0][1]
        
        # Final decision
        self.final_prediction = "Market is likely to go UP" if ensemble_pred else "Market is likely to go DOWN"
        self.confidence = ensemble_prob if ensemble_pred else 1 - ensemble_prob
        
        # Top features
        top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        self.top_features = {feature: importance for feature, importance in top_features}
        
        return self.final_prediction, self.confidence, self.top_features
    
    def run_analysis(self):
        return (self
                .fetch_data()
                .generate_indicators()
                .prepare_features()
                .build_models()
                .evaluate_models()
                .predict_market())

# Run the system
if __name__ == "__main__":
    # Initialize and run the system
    system = CompactFinancialAnalysisSystem(
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],  # Market ETFs
        lookback_years=3,
        macro_indicators=True,
        sentiment_analysis=True,
        feature_selection=True,
        n_features=50
    )
    
    # Run analysis
    prediction, confidence, top_features = system.run_analysis()
    
    # Print results
    print("\n" + "="*50)
    print(f"{prediction} (Confidence: {confidence:.2%})")
    print("="*50)
    
    # Print top influential features
    print("\nTop influential features:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Print model evaluation
    print("\nModel performance:")
    for model_name, metrics in system.evaluation_results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Final verdict with confidence
    print("\nFINAL VERDICT:")
    print(f"{prediction} (Confidence: {confidence:.2%})")