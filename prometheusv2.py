import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import datetime as dt
import warnings
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from statsmodels.tsa.stattools import adfuller
import ta
import textblob
from concurrent.futures import ThreadPoolExecutor
import finnhub
from fredapi import Fred  # For FRED data
warnings.filterwarnings('ignore')

# Compact financial analysis system class
class CompactFinancialAnalysisSystem:
    def __init__(self, symbols=['SPY', 'QQQ', 'DIA'], lookback_years=3, macro_indicators=True, sentiment_analysis=True, feature_selection=True, n_features=50):
        """Initialize the system with symbols, lookback period, and feature options."""
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
        # Initialize Finnhub client (replace with your API key)
        print("Initializing Finnhub client...")
        self.finnhub_client = finnhub.Client(api_key="cv9orrpr01qpd9s8hhggcv9orrpr01qpd9s8hhh0")  # Replace with your Finnhub API key
        # Initialize FRED client with provided API key
        print("Initializing FRED client with API key...")
        self.fred = Fred(api_key="3173a18ce8eb6fd417c310f8f887f3a5")

    def fetch_data(self):
        """Fetch market data for all symbols using yfinance."""
        print("Fetching market data for symbols:", self.symbols)
        with ThreadPoolExecutor(max_workers=10) as executor:
            self.data = {symbol: yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False) 
                        for symbol in self.symbols}
        
        if self.macro_indicators:
            print("Fetching macroeconomic data from FRED...")
            self.macro_data = self._fetch_real_macro_data()
        
        for symbol in self.symbols[:]:  # Copy to allow removal
            if self.data[symbol].empty:
                print(f"No data found for {symbol}. Removing from analysis.")
                self.symbols.remove(symbol)
            else:
                print(f"Filling missing data for {symbol}...")
                self.data[symbol].fillna(method='ffill', inplace=True)
                self.data[symbol].fillna(method='bfill', inplace=True)
        
        return self
    
    def _fetch_real_macro_data(self):
        """Fetch real macroeconomic indicators from FRED."""
        print("Fetching macroeconomic data from FRED...")
        
        # Economic growth indicators
        print("Fetching GDP data...")
        gdp = self.fred.get_series('GDP')  # Gross Domestic Product
        real_gdp = self.fred.get_series('GDPC1')  # Real Gross Domestic Product
        gdp_growth = gdp.pct_change(4)  # Quarterly GDP growth
        real_gdp_growth = real_gdp.pct_change(4)  # Real GDP growth
        
        # Inflation indicators
        print("Fetching inflation data...")
        cpi = self.fred.get_series('CPIAUCSL')  # Consumer Price Index
        inflation = cpi.pct_change(12)  # Annual inflation rate
        core_cpi = self.fred.get_series('CPILFESL')  # CPI Less Food and Energy
        core_inflation = core_cpi.pct_change(12)  # Core inflation rate
        breakeven_10y = self.fred.get_series('T10YIE')  # 10-Year Breakeven Inflation Rate
        
        # Interest rate indicators
        print("Fetching interest rate data...")
        fed_funds = self.fred.get_series('FEDFUNDS')  # Federal Funds Rate
        treasury_10y = self.fred.get_series('GS10')  # 10-Year Treasury Yield
        treasury_2y = self.fred.get_series('GS2')  # 2-Year Treasury Yield
        mortgage_30y = self.fred.get_series('MORTGAGE30US')  # 30-Year Fixed Rate Mortgage
        
        # Yield curve indicators
        print("Fetching yield curve data...")
        yield_curve_10y_2y = self.fred.get_series('T10Y2Y')  # 10Y-2Y Treasury Spread
        yield_curve_10y_3m = self.fred.get_series('T10Y3M')  # 10Y-3M Treasury Spread
        
        # Labor market indicators
        print("Fetching labor market data...")
        unemployment = self.fred.get_series('UNRATE')  # Unemployment Rate
        
        # Housing market indicators
        print("Fetching housing market data...")
        home_price_index = self.fred.get_series('CSUSHPISA')  # Case-Shiller Home Price Index
        median_house_price = self.fred.get_series('MSPUS')  # Median Sales Price of Houses
        
        # Monetary policy indicators
        print("Fetching monetary policy data...")
        m1 = self.fred.get_series('M1SL')  # M1 Money Supply
        m2 = self.fred.get_series('M2SL')  # M2 Money Supply
        fed_assets = self.fred.get_series('WALCL')  # Fed Total Assets
        reverse_repo = self.fred.get_series('RRPONTSYD')  # Overnight Reverse Repurchase Agreements
        
        # Market risk indicators
        print("Fetching market risk data...")
        high_yield_spread = self.fred.get_series('BAMLH0A0HYM2')  # ICE BofA US High Yield Index Spread
        
        # Process and combine data
        print("Processing and combining macroeconomic data...")
        macro_data = pd.DataFrame({
            # Economic growth
            'gdp_level': gdp,
            'real_gdp_level': real_gdp,
            'gdp_growth': gdp_growth,
            'real_gdp_growth': real_gdp_growth,
            
            # Inflation
            'cpi_level': cpi,
            'inflation': inflation,
            'core_inflation': core_inflation,
            'breakeven_inflation_10y': breakeven_10y,
            
            # Interest rates
            'fed_funds_rate': fed_funds / 100,  # Convert to decimal
            'treasury_10y': treasury_10y / 100,
            'treasury_2y': treasury_2y / 100,
            'mortgage_30y': mortgage_30y / 100,
            
            # Yield curve
            'yield_curve_10y_2y': yield_curve_10y_2y / 100,
            'yield_curve_10y_3m': yield_curve_10y_3m / 100,
            
            # Labor market
            'unemployment_rate': unemployment / 100,
            
            # Housing market
            'home_price_index': home_price_index,
            'median_house_price': median_house_price,
            
            # Monetary policy
            'm1_money_supply': m1,
            'm2_money_supply': m2,
            'm1_growth': m1.pct_change(12),
            'm2_growth': m2.pct_change(12),
            'fed_assets': fed_assets,
            'reverse_repo': reverse_repo,
            
            # Market risk
            'high_yield_spread': high_yield_spread / 100
        })
        
        # Handle frequencies and reindex to match stock data
        print("Aligning macroeconomic data with stock data frequency...")
        macro_data = macro_data.reindex(self.data[self.symbols[0]].index, method='ffill')
        
        # Calculate additional derived indicators
        print("Calculating derived macroeconomic indicators...")
        macro_data['real_interest_rate'] = macro_data['treasury_10y'] - macro_data['inflation']
        macro_data['fed_funds_real'] = macro_data['fed_funds_rate'] - macro_data['inflation']
        macro_data['mortgage_spread'] = macro_data['mortgage_30y'] - macro_data['treasury_10y']
        
        # Money supply / GDP ratio (proxy for liquidity)
        if not macro_data['gdp_level'].isna().all() and not macro_data['m2_money_supply'].isna().all():
            macro_data['m2_to_gdp'] = macro_data['m2_money_supply'] / macro_data['gdp_level'].ffill()
        
        # Calculate rates of change for key indicators
        for col in ['fed_funds_rate', 'treasury_10y', 'treasury_2y', 'unemployment_rate', 'home_price_index']:
            macro_data[f'{col}_3m_change'] = macro_data[col].diff(60)  # ~3 months of trading days
            macro_data[f'{col}_6m_change'] = macro_data[col].diff(120)  # ~6 months of trading days
        
        # Clean up data
        print("Cleaning up macroeconomic data...")
        macro_data = macro_data.replace([np.inf, -np.inf], np.nan)
        macro_data = macro_data.fillna(method='ffill')
        macro_data = macro_data.fillna(method='bfill')
        macro_data = macro_data.fillna(0)
        
        print("Macroeconomic data preparation completed.")
        return macro_data
    
    def fetch_real_sentiment(self, symbol):
        """Fetch real news sentiment from Finnhub."""
        print(f"Fetching real news sentiment for {symbol}...")
        try:
            news = self.finnhub_client.company_news(symbol, _from=self.start_date, to=self.end_date)
            if not news:
                print(f"No news data for {symbol}. Using neutral sentiment (0.0).")
                return pd.Series(0.0, index=self.data[symbol].index)
            sentiment_scores = [textblob.TextBlob(article['summary']).sentiment.polarity for article in news]
            dates = [pd.to_datetime(article['datetime'], unit='s') for article in news]
            # Aggregate duplicates by averaging sentiment for the same timestamp
            temp_df = pd.DataFrame({'sentiment': sentiment_scores}, index=dates)
            sentiment_df = temp_df.groupby(level=0).mean()['sentiment']  # Average sentiment per unique date
            sentiment_df = sentiment_df.reindex(self.data[symbol].index, method='nearest').fillna(method='ffill').fillna(0.0)
            print(f"News sentiment for {symbol} fetched successfully.")
            return sentiment_df
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}. Using neutral sentiment (0.0).")
            return pd.Series(0.0, index=self.data[symbol].index)
    
    def fetch_social_sentiment(self, symbol):
        """Fetch real social sentiment from Finnhub (Twitter/StockTwits)."""
        print(f"Fetching real social sentiment for {symbol}...")
        try:
            sentiment = self.finnhub_client.stock_social_sentiment(symbol, _from=self.start_date, to=self.end_date)
            if not sentiment.get('twitter', []):
                print(f"No social sentiment data for {symbol}. Using neutral sentiment (0.0).")
                return pd.Series(0.0, index=self.data[symbol].index)
            scores = [item['score'] for item in sentiment['twitter']]
            dates = [pd.to_datetime(item['at']) for item in sentiment['twitter']]
            sentiment_df = pd.Series(scores, index=dates).reindex(self.data[symbol].index, method='nearest').fillna(method='ffill').fillna(0.0)
            print(f"Social sentiment for {symbol} fetched successfully.")
            return sentiment_df
        except Exception as e:
            print(f"Error fetching social sentiment for {symbol}: {e}. Using neutral sentiment (0.0).")
            return pd.Series(0.0, index=self.data[symbol].index)
    
    def fetch_analyst_sentiment(self, symbol):
        """Fetch real analyst sentiment from Finnhub recommendation trends."""
        print(f"Fetching real analyst sentiment for {symbol}...")
        try:
            recommendations = self.finnhub_client.recommendation_trends(symbol)
            if not recommendations:
                print(f"No analyst recommendations for {symbol}. Using neutral sentiment (0.0).")
                return pd.Series(0.0, index=self.data[symbol].index)
            # Map recommendations to sentiment scores (buy=1, hold=0, sell=-1)
            sentiment_map = {'buy': 1, 'hold': 0, 'sell': -1, 'strong buy': 1, 'strong sell': -1}
            scores = [sentiment_map.get(rec.get('buy', 'hold').lower(), 0) for rec in recommendations]
            dates = [pd.to_datetime(rec['period']) for rec in recommendations]
            sentiment_df = pd.Series(scores, index=dates).reindex(self.data[symbol].index, method='nearest').fillna(method='ffill').fillna(0.0)
            print(f"Analyst sentiment for {symbol} fetched successfully.")
            return sentiment_df
        except Exception as e:
            print(f"Error fetching analyst sentiment for {symbol}: {e}. Using neutral sentiment (0.0).")
            return pd.Series(0.0, index=self.data[symbol].index)
    
    def generate_indicators(self):
        """Generate technical, macro, and sentiment indicators for all symbols."""
        print("Generating indicators for all symbols...")
        for symbol in self.symbols:
            print(f"Processing indicators for {symbol}...")
            df = self.data[symbol].copy()
            # Ensure Series type by using iloc[:, 0] if DataFrame
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
            volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
            high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
            low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
            
            self.indicators[symbol] = pd.DataFrame(index=df.index)
            
            # Basic returns and volume changes
            self.indicators[symbol]['return_1d'] = close.pct_change(1)
            self.indicators[symbol]['return_5d'] = close.pct_change(5)
            self.indicators[symbol]['return_10d'] = close.pct_change(10)
            self.indicators[symbol]['return_20d'] = close.pct_change(20)
            self.indicators[symbol]['return_60d'] = close.pct_change(60)
            self.indicators[symbol]['vol_change_1d'] = volume.pct_change(1)
            self.indicators[symbol]['vol_change_5d'] = volume.pct_change(5)
            self.indicators[symbol]['price_vol_corr_20d'] = close.rolling(20).corr(volume)
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
            
            # ATR
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            for window in [7, 14, 21]:
                self.indicators[symbol][f'atr_{window}'] = true_range.rolling(window).mean()
                self.indicators[symbol][f'atr_pct_{window}'] = self.indicators[symbol][f'atr_{window}'] / close
            
            # ADX
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
            
            # VWAP
            df['vwap_daily'] = ((close + high + low) / 3 * volume).cumsum() / volume.cumsum()
            self.indicators[symbol]['vwap_ratio'] = close / df['vwap_daily']
            
            # OBV
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            self.indicators[symbol]['obv'] = obv
            for window in [10, 20, 50]:
                self.indicators[symbol][f'obv_ma_{window}'] = obv.rolling(window).mean()
                self.indicators[symbol][f'obv_ratio_{window}'] = obv / self.indicators[symbol][f'obv_ma_{window}']
            
            # Ichimoku Cloud
            high_9 = high.rolling(9).max()
            low_9 = low.rolling(9).min()
            self.indicators[symbol]['ichimoku_conversion'] = (high_9 + low_9) / 2
            high_26 = high.rolling(26).max()
            low_26 = low.rolling(26).min()
            self.indicators[symbol]['ichimoku_base'] = (high_26 + low_26) / 2
            self.indicators[symbol]['ichimoku_span_a'] = ((self.indicators[symbol]['ichimoku_conversion'] + self.indicators[symbol]['ichimoku_base']) / 2).shift(26)
            self.indicators[symbol]['ichimoku_span_b'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            
            # Additional TA indicators
            self.indicators[symbol]['cci_20'] = ta.trend.cci(high, low, close, window=20)
            self.indicators[symbol]['roc_20'] = ta.momentum.roc(close, window=20)
            self.indicators[symbol]['stoch_k'] = ta.momentum.stoch(high, low, close)
            self.indicators[symbol]['stoch_d'] = ta.momentum.stoch_signal(high, low, close)
            self.indicators[symbol]['williamsr_14'] = ta.momentum.williams_r(high, low, close, lbp=14)
            
            # Statistical features
            for window in [10, 20, 50]:
                self.indicators[symbol][f'return_skew_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).skew()
                self.indicators[symbol][f'return_kurt_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).kurt()
                self.indicators[symbol][f'return_vol_{window}'] = self.indicators[symbol]['return_1d'].rolling(window).std()
                self.indicators[symbol][f'return_z_{window}'] = (self.indicators[symbol]['return_1d'] - self.indicators[symbol]['return_1d'].rolling(window).mean()) / self.indicators[symbol]['return_1d'].rolling(window).std()
                mean_return = self.indicators[symbol]['return_1d'].rolling(window).mean() * 252
                std_return = self.indicators[symbol]['return_1d'].rolling(window).std() * np.sqrt(252)
                self.indicators[symbol][f'sharpe_{window}'] = mean_return / std_return
            
            # Macro indicators
            if self.macro_indicators and hasattr(self, 'macro_data'):
                print(f"Adding macroeconomic indicators for {symbol}...")
                aligned_macro_data = self.macro_data.reindex(self.indicators[symbol].index).interpolate(method='linear')
                for col in aligned_macro_data.columns:
                    self.indicators[symbol][f'macro_{col}'] = aligned_macro_data[col]
                    self.indicators[symbol][f'macro_{col}_roc_20'] = aligned_macro_data[col].pct_change(20)
                    self.indicators[symbol][f'macro_{col}_roc_60'] = aligned_macro_data[col].pct_change(60)
            
            # Sentiment indicators
            if self.sentiment_analysis:
                print(f"Adding sentiment indicators for {symbol}...")
                self.indicators[symbol]['news_sentiment'] = self.fetch_real_sentiment(symbol)
                #self.indicators[symbol]['social_sentiment'] = self.fetch_social_sentiment(symbol)
                self.indicators[symbol]['analyst_sentiment'] = self.fetch_analyst_sentiment(symbol)
                for col in ['news_sentiment', 'analyst_sentiment']:
                    self.indicators[symbol][col] = np.clip(self.indicators[symbol][col], -1, 1)
                    for window in [5, 10, 20]:
                        self.indicators[symbol][f'{col}_ma_{window}'] = self.indicators[symbol][col].rolling(window).mean()
                self.indicators[symbol]['combined_sentiment'] = (self.indicators[symbol]['news_sentiment'] + 
                                                                self.indicators[symbol]['analyst_sentiment']) / 2
            
            # Clean up
            print(f"Cleaning up indicators for {symbol} (handling NaN, inf)...")
            self.indicators[symbol] = self.indicators[symbol].replace([np.inf, -np.inf], np.nan)
            self.indicators[symbol].fillna(method='ffill', inplace=True)
            self.indicators[symbol].fillna(method='bfill', inplace=True)
            self.indicators[symbol].fillna(0, inplace=True)
        
        print("Indicator generation completed for all symbols.")
        return self
    
    def prepare_features(self):
        """Prepare feature matrix for modeling."""
        print("Preparing feature matrix...")
        all_features = pd.DataFrame()
        for symbol in self.symbols:
            print(f"Combining indicators for {symbol} into feature matrix...")
            indicators = self.indicators[symbol].copy()
            indicators.columns = [f'{symbol}_{col}' if col != 'target' else col for col in indicators.columns]
            if all_features.empty:
                all_features = indicators
            else:
                indicators = indicators.drop('target', axis=1, errors='ignore')
                all_features = all_features.join(indicators)
        
        print("Trimming initial rows to ensure indicator stability (200 days)...")
        all_features = all_features.iloc[200:].copy()
        self.features = [col for col in all_features.columns if col != 'target']
        
        if self.feature_selection and len(self.features) > self.n_features:
            print(f"Performing feature selection: selecting top {self.n_features} features...")
            X = all_features[self.features]
            y = all_features['target']
            selector = SelectKBest(f_classif, k=self.n_features)
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            self.features = [self.features[i] for i in selected_indices]
            self.feature_importances = {self.features[i]: selector.scores_[selected_indices[i]] for i in range(len(self.features))}
        
        print("Scaling features with StandardScaler...")
        X = all_features[self.features]
        y = all_features['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        self.feature_matrix = pd.DataFrame(X_scaled, index=all_features.index, columns=self.features)
        self.target = all_features['target']
        print("Feature preparation completed.")
        return self
    
    def build_models(self):
        """Build and train ensemble and individual models."""
        print("Building and training machine learning models...")
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
        ]
        self.models['ensemble'] = VotingClassifier(estimators=base_models, voting='soft')
        print("Training ensemble model...")
        self.models['ensemble'].fit(self.X_train, self.y_train)
        for name, model in base_models:
            print(f"Training {name} model...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        if 'rf' in self.models:
            self.feature_importances = {feature: importance for feature, importance in zip(self.features, self.models['rf'].feature_importances_)}
        print("All models trained successfully.")
        return self
    
    def evaluate_models(self):
        """Evaluate model performance on test data."""
        print("Evaluating model performance...")
        results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name} model...")
            y_pred = model.predict(self.X_test)
            results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
        self.evaluation_results = results
        print("Model evaluation completed.")
        return self
    
    def predict_market(self):
        """Make predictions for the latest data point."""
        print("Making market prediction for the latest data...")
        latest_data = self.feature_matrix.iloc[-1:].values
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(latest_data)[0]
            prob = model.predict_proba(latest_data)[0][1] if hasattr(model, 'predict_proba') else 0.5
            predictions[name] = (pred, prob)
        
        ensemble_pred = self.models['ensemble'].predict(latest_data)[0]
        ensemble_prob = self.models['ensemble'].predict_proba(latest_data)[0][1]
        
        self.final_prediction = "Market is likely to go UP" if ensemble_pred else "Market is likely to go DOWN"
        self.confidence = ensemble_prob if ensemble_pred else 1 - ensemble_prob
        
        top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        self.top_features = {feature: importance for feature, importance in top_features}
        
        print(f"Prediction: {self.final_prediction} with confidence {self.confidence:.2%}")
        return self.final_prediction, self.confidence, self.top_features
    
    def run_analysis(self):
        """Run the full analysis pipeline."""
        print("Starting full market analysis...")
        return (self
                .fetch_data()
                .generate_indicators()
                .prepare_features()
                .build_models()
                .evaluate_models()
                .predict_market())

# Run the system
if __name__ == "__main__":
    print("Initializing CompactFinancialAnalysisSystem...")
    system = CompactFinancialAnalysisSystem(
        symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
        lookback_years=3,
        macro_indicators=True,
        sentiment_analysis=True,
        feature_selection=True,
        n_features=50
    )
    print("Running analysis...")
    prediction, confidence, top_features = system.run_analysis()
    
    # Display results
    print("\n" + "="*50)
    print(f"{prediction} (Confidence: {confidence:.2%})")
    print("="*50)
    
    print("\nTop influential features:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    print("\nModel performance:")
    for model_name, metrics in system.evaluation_results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    print("\nFINAL VERDICT:")
    print(f"{prediction} (Confidence: {confidence:.2%})")