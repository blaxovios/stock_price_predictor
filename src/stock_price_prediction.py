import os
from tqdm import tqdm
import torch
from typing import Union, Tuple
import logging
import pandas as pd
import polars as pl
import numpy as np
import glob
from enum import Enum
import spacy
from multiprocessing import cpu_count
import joblib
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
# Import local modules
from utils.utils import setup_logging


# Setup logging and download required NLTK lexicon.
setup_logging()


class ScrapedDataCategory(Enum):
    NEWS = "news"
    PRICES = "prices"
    
class StockSymbol(Enum):
    NVDA = "NVDA"
    
class ColumnNames(Enum):
    PRICE_CHANGE_PREDICTION = "price_change_prediction"
    CLOSE_PRICE_PCT_CHANGE = "close_price_pct_change"


class PreprocessNewsData:
    def __init__(self):
        self.stock_symbols_json = 'data/static/exports/json/scrapers/stocks/symbols.json'
        self.num_cores = cpu_count()
        
    def merge_parquet_data(self, directory: str, data_category: str = ScrapedDataCategory.NEWS.value, export_to_parquet: bool = True, export_dir: str = 'data/processed') -> str:
        """
        Merge parquet data from a directory of parquet files into a single LazyFrame,
        and optionally export it to a parquet file.
        
        The merged dataframe is exported to a parquet file with the same columns as the
        input dataframes, and the same order of columns. The rows of the merged dataframe
        are ordered in the same order as the input dataframes.
        
        Parameters
        ----------
        directory : str
            The directory containing the parquet files to be merged.
        data_category : str, optional
            The category of the data to be merged. Either 'news' or 'prices'.
        export_to_parquet : bool, optional
            If True, the merged dataframe is exported to a parquet file.
            If False, the merged dataframe is returned as a LazyFrame.
        
        Returns
        -------
        str
            The path to the exported parquet file if export_to_parquet is True, otherwise None.
        """
        logging.info(f"Merging {data_category} data...")
        merged_dfs = []
        for file in glob.glob(directory):
            file_size = os.path.getsize(file)
            if file_size > 12:
                merged_dfs.append(pl.scan_parquet(file))
        df = pl.concat(merged_dfs, how='vertical_relaxed')
        export_path = export_dir + f"/{data_category}.parquet"
        if export_to_parquet:
            df.sink_parquet(export_path)
        return export_path
    
    def merge_scraped_news_with_prices_data(self, news_df: pd.DataFrame, stock_prices_df: pd.DataFrame, export_path: str = None, export_to_parquet: bool = True) -> pd.DataFrame:
        """
        Merge news data with stock prices data based on the article date and stock price date, respectively.
        
        The merged dataframe will contain the news data with the corresponding stock prices data.
        
        The article date and stock price date will be converted to date type before merging.
        
        The merged dataframe will be sorted by article date and set as the index.
        
        Any rows with missing article dates will be dropped.
        
        Parameters
        ----------
        news_df : pd.DataFrame
            The news data to be merged with the stock prices data.
        stock_prices_df : pd.DataFrame
            The stock prices data to be merged with the news data.
        
        Returns
        -------
        pd.DataFrame
            The merged dataframe with the news data and stock prices data.
        """
        # Merge Scraped Data. Convert to date because the article_date column has a datetime64[ms] dtype,
        # which means it has millisecond precision, while the stock_price_date column has a datetime64[ns] dtype,
        # which means it has nanosecond precision.
        news_df['article_date'] = news_df['article_date'].dt.date
        stock_prices_df['stock_price_date'] = stock_prices_df['stock_price_date'].dt.date
        merged_data = pd.merge(news_df, stock_prices_df, left_on='article_date', right_on='stock_price_date', how='left')
        
        # Rename columns ending with _x to _news and _y to _stock_prices
        merged_data.columns = [col.replace('_x', '_news').replace('_y', '_stock_prices') for col in merged_data.columns]

        merged_data.dropna(subset=['article_date'], inplace=True)
        merged_data.sort_values(by='article_date', inplace=True)
        merged_data.set_index('article_date', inplace=True)
        merged_data.index = pd.to_datetime(merged_data.index)
        # Because stock prices dataset is updated daily per business day, while news data is updated daily by calendar. Use ffill to fill missing stock prices with previous day value.
        merged_data.ffill(inplace=True)
        if export_to_parquet and export_path:
            merged_data.to_parquet(export_path)
        return merged_data
        
    def preprocess_scraped_news_data(self, filepath: str) -> pd.DataFrame:
        """
        Preprocesses scraped news data from a parquet file and returns a new LazyFrame.
        
        The preprocessing steps include:
        - Keeping unique rows based on the 'id' column.
        - Keeping rows where 'content' is not empty or contains only whitespace or a single dash "-".
        - Dropping rows where article_date is null.
        - Converting stock_prices field: strip (%) and cast to float.
        - Converting 'article_date' from datetime to date (yyyy-mm-dd) while keeping the latest.
        """
        logging.info("Preprocessing scraped news dataframe with polars...")
        df = pl.scan_parquet(filepath)
        # Keep unique rows based on the 'id' column.
        df = df.unique(subset=["id"], maintain_order=True)
        # Keep rows where 'content' is not empty or contains only whitespace or a single dash "-".
        df = df.filter(
            (pl.col("content").str.strip_chars().ne("")) & 
            (pl.col("content").is_not_null()) & 
            (pl.col("content").str.strip_chars().ne("-"))
            )
        # Drop rows where article_date is null.
        df = df.filter(
            pl.col("article_date").cast(pl.Datetime, strict=False).is_not_null()
        )
        # Convert stock_prices field: strip (%) and cast to float.
        # df = df.filter(
        #     pl.col("stock_prices").map_elements(
        #         lambda s: {k: float(v.strip('(),.%').replace('+', '').replace(',', '').replace('(', '-').replace(')', '').replace('%', '')) if v else None for k, v in s.items()},
        #         return_dtype=pl.Struct
        #     ).is_not_null()
        # )
        
        # Drop column stock_prices from lazyframe
        df = df.drop("stock_prices")
        
        # Convert 'article_date' from datetime to date (yyyy-mm-dd) while keeping the latest
        df = df.with_columns(
            pl.col("article_date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
            .cast(pl.Datetime("ms"))
            .alias("article_date")
        ).sort("article_date", descending=True).unique(subset=["url", "article_date"], keep="last")
        
        df = df.collect().to_pandas()
        df['article_date'] = pd.to_datetime(df['article_date'])
        return df
    
    def preprocess_scraped_stock_prices_data(self, filepath: str) -> pd.DataFrame:
        logging.info("Preprocessing scraped stock prices dataframe with polars...")
        df = pl.scan_parquet(filepath)
        df = df.with_columns([
            pl.col("stock_price_date").str.strptime(pl.Datetime, format="%b %d, %Y").dt.strftime("%Y-%m-%d %H:%M:%S"),
            pl.col("open_price").cast(pl.Float64),
            pl.col("high_price").cast(pl.Float64),
            pl.col("low_price").cast(pl.Float64),
            pl.col("close_price").cast(pl.Float64),
            pl.col("adj_close_price").cast(pl.Float64),
            pl.col("volume").str.replace_all(",", "").cast(pl.Float64)
        ])
        
        df = df.collect().to_pandas()
        df['stock_price_date'] = pd.to_datetime(df['stock_price_date'])
        # Calculate the percentage change for the stock price for current day, compared to previous business day
        df = df.sort_values(by=['stock_price_date', 'symbol'])
        df['prev_close_price'] = df['close_price'].shift().where(df['symbol'] == df['symbol'].shift(), np.nan)
        df[ColumnNames.CLOSE_PRICE_PCT_CHANGE.value] = (df['close_price'] - df['prev_close_price']) / df['prev_close_price']
        df = df.drop(columns=['prev_close_price'])
        return df
        
    def load_news_df(self, data_path: str, n_rows: int = None) -> pd.DataFrame:
        """
        Load a parquet file containing news data.

        Args:
            data_path (str): File path to the parquet file to load.
            n_rows (int, optional): Number of rows to load from the top of the dataframe, sorted in descending order by article_date. Defaults to None.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """
        logging.info(f"Loading news data from {data_path}...")
        if n_rows:
            df = pd.read_parquet(data_path).sort_values(by='article_date', ascending=False).head(n_rows)
        else:
            df = pd.read_parquet(data_path)
        
        return df
    
    
class ClassifyNewsData:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.n_threads = cpu_count()
        self.n_topics = 10
                  
    def _clean_key(self, key: str) -> str:
        """
        Clean a key by removing any special characters.

        Args:
            key (str): The key to clean.

        Returns:
            str: The cleaned key.
        """
        return key.replace('^', '').replace('[', '').replace(']', '').replace('<', '')

    def clean_keys(self, d: dict, keys_to_extract: list[str] = None) -> dict[str, str]:
        """
        Clean a dictionary of keys by removing any special characters and extracting specific keys.

        Args:
            d (dict): The dictionary to clean.
            keys_to_extract (list[str], optional): The list of keys to extract and clean. Defaults to None.

        Returns:
            dict[str, str]: A new dictionary with cleaned and extracted keys.
        """
        new_dict = {}
        if keys_to_extract is None:
            keys_to_extract = list(d.keys())
        for k in keys_to_extract:
            if k in d:
                ck = self._clean_key(k)
                new_dict[ck] = d[k]
        return new_dict
         
    def preprocess_df(self, df: pd.DataFrame, export_to_parquet: bool = True, export_dir: str = 'data/processed') -> pd.DataFrame:
        """
        Preprocesses a DataFrame by extracting stock prices, cleaning keys, and running LDA with a Random Forest classifier
        to predict class and sentiment.

        Args:
            df (pd.DataFrame): DataFrame containing the news content and price change data.
            target_column (str): Name of the column containing the price change data.
            export_to_parquet (bool, optional): Whether to export the preprocessed DataFrame to a parquet file. Defaults to True.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with predicted class and sentiment.
        """
        logging.info("Preprocessing text with spacy...")
        # TODO: May need to enable 'tagger'+'attribute_ruler' or 'morphologizer'.
        docs = self.nlp.pipe(texts=df['content'], batch_size=100, disable=["ner", "parser", "tagger", "entity_linker", "entity_ruler", "textcat", "textcat_multilabel", "morphologizer", "attribute_ruler", "senter", "sentencizer", "transformer"])
        sequences = (
            [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ and not token.lemma_.isspace()]
            for doc in docs
        )
        df['processed_text'] = list(
            tqdm(sequences, total=len(df), desc="Processing text")
        )
        
        # NOTE: See more: https://medium.com/@sayahfares19/text-analysis-topic-modelling-with-spacy-gensim-4cd92ef06e06
        logging.info("Creating bigrams...")
        # Treating certain word pairs as single tokens can improve the interpretability of our topics.
        # For example, “New York” is more meaningful as a single entity than “New” and “York” separately.
        # We can use Gensim’s Phrases model to detect common bigrams
        bigram = models.phrases.Phrases(df['processed_text'])
        df['processed_text'] = [bigram[line] for line in df['processed_text']]
        
        # TODO: Try different n_topics, models instead of LDA etc.
        logging.info("Topic modeling using LDA...")
        # Gensim requires two main components for topic modeling:
        # A Dictionary: mapping between words and their integer ids
        # A Corpus: a list of documents, where each document is represented as a bag-of-words
        dictionary = Dictionary(df['processed_text'])
        corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
        lda_model = LdaModel(corpus=corpus, num_topics=self.n_topics, id2word=dictionary)
        df['doc_topics'] = [lda_model.get_document_topics(bow) for bow in corpus]
        df['doc_weight'] = df['doc_topics'].apply(lambda td: sum(weight for _, weight in td))
        daily = (df.groupby(df.index.date)['doc_weight'].sum().rename('daily_weight'))

        # min–max normalization into [0,1]
        min_w, max_w = daily.min(), daily.max()
        if max_w > min_w:
            daily = (daily - min_w) / (max_w - min_w)
        else:
            daily[:] = 0.0
        # merge back onto your original df (every row for a given date
        # gets the same normalized daily score)
        df = df.join(daily, on=df.index.date)
        df.rename(columns={'daily_weight': 'news_weight_on_stock_price_change'}, inplace=True)

        cols_to_drop = ['key_0', 'doc_topics', 'doc_weight', 'id_news', 'id_stock_prices', 'url_stock_prices', 'url_news', 'title', 'content', 'timestamp_news', 'timestamp_stock_prices', 'sentiment']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
        
        df = df.loc[~df.index.normalize().duplicated(keep='first')].copy()
        if export_to_parquet:
            df.to_parquet(path=''.join([export_dir, '/processed_news_data.parquet']), index=True)
        logging.info("Dataframe preprocessed.")
        return df


class StockPricePredictor:
    def __init__(self, predictor_path: str, use_gpu: bool = True):
        """
        Initialize the predictor.

        Parameters
        ----------
        predictor_path : str
            Path to save/load the trained model.
        use_gpu : bool
            Whether to attempt to use GPU if available.
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.predictor_model_path = predictor_path
        self.best_params_ = None

    def create_features(
        self,
        df: pd.DataFrame,
        label: str = None,
        news_df: pd.DataFrame = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Build feature matrix, optionally shifting target and including news.

        - Extracts datetime features.
        - Merges daily news sentiment/weights if provided, avoiding column collisions.
        - Shifts target by -1 to predict next day.

        Returns X, y (if label provided) or X only.
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # 1) Datetime features
        df['hour']       = df.index.hour
        df['dayofweek']  = df.index.dayofweek
        df['quarter']    = df.index.quarter
        df['month']      = df.index.month
        df['year']       = df.index.year
        df['dayofyear']  = df.index.dayofyear
        df['dayofmonth'] = df.index.day

        feature_cols = [
            'hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth'
        ]

        # 2) Merge news signals, if any
        if news_df is not None:
            daily = news_df.copy()
            daily.index = pd.to_datetime(daily.index)
            daily = daily.resample('D').mean()

            # drop any overlapping cols so join won’t error
            overlap = daily.columns.intersection(df.columns)
            if len(overlap):
                df = df.drop(columns=overlap)

            df = df.join(daily, how='left')
            feature_cols += list(daily.columns)

        X = df[feature_cols]

        # 3) If we have a label, shift it to t+1
        if label:
            df['target_next'] = df[label].shift(-1)
            # drop the LAST row (because its target is NaN)
            X = X.iloc[:-1]
            y = df['target_next'].iloc[:-1]
            return X, y

        return X

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        test_size: int = 1,
        val_size: int = 30
    ):
        """
        Split into train/validation/test by integer indices.

        - Last `test_size` rows -> test
        - Preceding `val_size` rows -> validation
        - Rest -> train

        Returns (X_train, y_train), (X_val, y_val), (X_test, y_test).
        """
        if y is not None:
            assert len(X) == len(y), "Features and target must align"
        n = len(X)
        i_test_start = n - test_size
        i_val_start  = i_test_start - val_size

        X_train = X.iloc[:i_val_start]
        X_val   = X.iloc[i_val_start:i_test_start]
        X_test  = X.iloc[i_test_start:]

        if y is not None:
            y_train = y.iloc[:i_val_start]
            y_val   = y.iloc[i_val_start:i_test_start]
            y_test  = y.iloc[i_test_start:]
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)

        return X_train, X_val, X_test

    def train_model(
        self,
        df: pd.DataFrame,
        target_column: str,
        news_df: pd.DataFrame = None
    ) -> XGBRegressor:
        """
        Train XGBRegressor with time-series CV on train+val, evaluate on test.
        Saves best model to disk.
        """
        logging.info("Preparing data for training...")
        X, y = self.create_features(df, label=target_column, news_df=news_df)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.split_data(X, y, test_size=1, val_size=30)

        # Combine train+val for CV
        X_tv = pd.concat([X_train, X_val])
        y_tv = pd.concat([y_train, y_val])

        cv = TimeSeriesSplit(n_splits=4)
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 300, 500],
            'colsample_bytree': [0.5, 0.7]
        }

        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=1)
        logging.info("Starting grid search CV...")
        grid.fit(X_tv, y_tv)
        best = grid.best_estimator_
        logging.info(f"Best params: {grid.best_params_}")

        # Final evaluation on test set
        preds = best.predict(X_test)
        mse  = mean_squared_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        ae = abs(y_test.iloc[0] - preds[0])
        pe = ae / abs(y_test.iloc[0])
        se = (y_test.iloc[0] - preds[0])**2
        
        logging.info(f"Test MSE: {mse:.4f}, R2: {r2:.4f}, AE: {ae:.4f}, PE: {pe:.4f}, SE: {se:.4f}")

        # Save model
        os.makedirs(os.path.dirname(self.predictor_model_path), exist_ok=True)
        joblib.dump(best, self.predictor_model_path)
        logging.info(f"Model saved to {self.predictor_model_path}")
        return best

    def load_model(self) -> XGBRegressor:
        """
        Load the trained model from disk.
        """
        if not os.path.exists(self.predictor_model_path):
            raise FileNotFoundError(f"No model found at {self.predictor_model_path}")
        model = joblib.load(self.predictor_model_path)
        logging.info(f"Model loaded from {self.predictor_model_path}")
        return model

    def predict(
        self,
        df: pd.DataFrame,
        model: XGBRegressor = None,
        target_column: str = None,
        news_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate predictions for next-day change and return DataFrame with results.

        If `model` is None, will load from disk. Requires `target_column` to shift features.
        """
        if model is None:
            model = self.load_model()
        if target_column is None:
            raise ValueError("`target_column` must be provided for predict().")

        X, y = self.create_features(df, label=target_column, news_df=news_df)
        preds = model.predict(X)
        result = df.iloc[1:].copy()
        result['predicted_change'] = preds
        return result
    
    def backtest(
        self,
        df: pd.DataFrame,
        target_column: str,
        news_df: pd.DataFrame = None,
        initial_train_size: int = 60
    ) -> pd.DataFrame:
        """
        Perform an expanding-window backtest: train on data up to each day,
        then predict the next day's change. Returns DataFrame with columns
        ['true', 'predicted'] indexed by prediction date.
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        preds, truths, dates = [], [], []

        for i in range(initial_train_size, len(df)-1):
            train_df = df.iloc[:i]
            # train simplified model with best_params_
            if self.best_params_:
                mdl = XGBRegressor(objective='reg:squarederror', random_state=42, **self.best_params_)
            else:
                mdl = XGBRegressor(objective='reg:squarederror', random_state=42)

            X_train, y_train = self.create_features(
                train_df, label=target_column,
                news_df=(news_df.iloc[:i] if news_df is not None else None)
            )
            mdl.fit(X_train, y_train)

            # prepare next-day features
            window = df.iloc[:i+1]
            X_window = self.create_features(
                window, label=None,
                news_df=(news_df.iloc[:i+1] if news_df is not None else None)
            )
            pred = mdl.predict(X_window)[-1]
            true = df[target_column].iloc[i+1]
            date = df.index[i+1]

            preds.append(pred)
            truths.append(true)
            dates.append(date)

        return pd.DataFrame({'true': truths, 'predicted': preds}, index=dates)

    def evaluate_backtest(self, backtest_df: pd.DataFrame) -> dict:
        """
        Given a backtest DataFrame with 'true' and 'predicted', compute metrics.
        Returns a dict: mse, rmse, mae, mape, r2.
        """
        y_true = backtest_df['true']
        y_pred = backtest_df['predicted']
        
        mse  = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        
        # Compute SMAPE safely
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        # avoid dividing by zero
        nonzero = denom != 0
        smape = np.mean( np.abs(y_pred[nonzero] - y_true[nonzero]) / denom[nonzero] ) * 100

        # Also compute classic MAPE skipping zeros
        mask = y_true != 0
        mape = np.mean( np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]) ) * 100 if mask.any() else np.nan
        
        baseline = backtest_df['true'] * 0
        mse_baseline  = mean_squared_error(backtest_df['true'], baseline)
        rmse_baseline = root_mean_squared_error(backtest_df['true'], baseline)
        mae_baseline  = mean_absolute_error(backtest_df['true'], baseline)

        logging.info(f"Baseline RMSE: {rmse_baseline:.4f}, MAE: {mae_baseline:.4f}, MSE: {mse_baseline:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape (%)': mape,
            'smape (%)': smape,
            'r2': r2
        }

    def run_stock_price_predictive_pipeline(
        self,
        raw_scraped_news_parquet_dir_path: str = "data/scraped/news/*.parquet",
        raw_scraped_stock_prices_parquet_dir_path: str = "data/scraped/prices/*.parquet",
        merged_data_path: str = 'data/processed/merged_raw_scraped_news_with_stock_prices.parquet',
        processed_merged_data_path: str = "data/processed/processed_news_data.parquet",
        predictor_model_path: str = 'models/stock_price_predictor_forecaster.pkl',
        use_processed_data: bool = True,
        use_pretrained_model: bool = True,
        merge_and_process_raw_scraped_data: bool = False,
        target_column: str = ColumnNames.CLOSE_PRICE_PCT_CHANGE.value,
    ):
        """
        Complete pipeline to merge/load data, train or load model, predict next-day price change.
        Returns a DataFrame with a `predicted_change` column indexed by date.
        """
        # 1) Instantiate preprocessors and predictor
        pnd = PreprocessNewsData()
        cnd = ClassifyNewsData()
        spp = StockPricePredictor(predictor_path=predictor_model_path, use_gpu=True)

        # 2) Merge or load the raw scraped data
        if merge_and_process_raw_scraped_data:
            # Merge & preprocess news
            news_parquet = pnd.merge_parquet_data(
                directory=raw_scraped_news_parquet_dir_path,
                data_category=ScrapedDataCategory.NEWS.value
            )
            news_raw_df = pnd.preprocess_scraped_news_data(news_parquet)

            # Merge & preprocess stock prices
            prices_parquet = pnd.merge_parquet_data(
                directory=raw_scraped_stock_prices_parquet_dir_path,
                data_category=ScrapedDataCategory.PRICES.value
            )
            prices_raw_df = pnd.preprocess_scraped_stock_prices_data(prices_parquet)

            # Join news + prices
            merged_df = pnd.merge_scraped_news_with_prices_data(
                news_df=news_raw_df,
                stock_prices_df=prices_raw_df,
                export_path=merged_data_path
            )
        else:
            merged_df = pnd.load_news_df(data_path=merged_data_path)

        # Ensure index is datetime trading date
        merged_df['stock_price_date'] = pd.to_datetime(merged_df['stock_price_date'])
        merged_df.set_index('stock_price_date', inplace=True)
        merged_df.sort_index(inplace=True)

        # 3) Optionally classify / further preprocess
        if not use_processed_data:
            processed_df = cnd.preprocess_df(merged_df)
        else:
            processed_df = pd.read_parquet(processed_merged_data_path)
            processed_df.index = pd.to_datetime(processed_df.index)

        # 4) Identify numeric news features
        news_feature_cols = []
        if 'news_weight_on_stock_price_change' in processed_df.columns:
            news_feature_cols.append('news_weight_on_stock_price_change')
        if 'sentiment' in processed_df.columns:
            news_feature_cols.append('sentiment')
        if not news_feature_cols:
            raise ValueError(
                "No numeric news feature columns found. "
                "Expected at least 'news_weight_on_stock_price_change'."
            )
        news_df = processed_df[news_feature_cols]

        # 5) Check target exists
        if target_column not in processed_df.columns:
            raise ValueError(f"Target column `{target_column}` not found in DataFrame.")

        # 6) Train or load the model
        if not use_pretrained_model:
            model = spp.train_model(
                df=processed_df,
                target_column=target_column,
                news_df=news_df
            )
        else:
            model = spp.load_model()

        # 7) Predict next-day changes across the entire history
        results_df = spp.predict(
            df=processed_df,
            model=model,
            target_column=target_column,
            news_df=news_df
        )

        # 8) Log the latest prediction
        latest_date = results_df.index.max()
        latest_pred = results_df.loc[latest_date, 'predicted_change'] * 100
        direction = "up" if latest_pred > 0 else "down"
        logging.info(
            f"Latest prediction for {latest_date.date()}: "
            f"{latest_pred:.2f}% ({direction})"
        )

        return results_df

    def predict_next_day_pct_change_from_news(
        self,
        daily_news_df: pd.DataFrame,
        predictor_model_path: str,
        news_cols: list = None,
        use_gpu: bool = False
    ) -> float:
        """
        Given a DataFrame of daily news metrics (indexed by date),
        loads the saved model and returns the prediction for the next trading day.

        Parameters
        ----------
        daily_news_df : pd.DataFrame
            Must have a DateTimeIndex (one row per calendar day) and
            contain the numeric news features you want to use.
        predictor_model_path : str
            Path to the joblib (.pkl) file that was saved by train_model().
        news_cols : list of str, optional
            Column names in `daily_news_df` to treat as news features.
            If None, will automatically use all numeric columns in the DF.
        use_gpu : bool
            Whether to initialize the predictor with GPU support.

        Returns
        -------
        float
            Predicted next‐day percent change (e.g. 0.012 = +1.2%).
        """

        # 1) Copy & ensure index is datetime
        df = daily_news_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'stock_price_date' in df.columns:
                df['stock_price_date'] = pd.to_datetime(df['stock_price_date'])
                df.set_index('stock_price_date', inplace=True)
            else:
                raise ValueError("daily_news_df must have a DatetimeIndex or a 'stock_price_date' column.")
        df = df.sort_index()

        # 2) Decide which columns are your news features
        if news_cols is None:
            # take all numeric columns
            news_cols = df.select_dtypes(include='number').columns.tolist()
        missing = set(news_cols) - set(df.columns)
        if missing:
            raise KeyError(f"These news_cols are missing from your DataFrame: {missing}")
        news_df = df[news_cols]

        # 3) Load the predictor
        spp = StockPricePredictor(predictor_path=predictor_model_path, use_gpu=use_gpu)
        model = spp.load_model()

        # 4) Build features (time + your news)
        # Note: label=None means we just get X, no shifting of y
        X = spp.create_features(df=df, label=None, news_df=news_df)

        # 5) Predict on every day in X; last entry = next‐day forecast
        preds = model.predict(X)

        # 6) Return the final one
        return float(preds[-1])
            
            
if __name__ == "__main__":
    # --- CONFIG ---
    raw_scraped_news_parquet_dir_path   = "data/scraped/news/*.parquet"
    raw_scraped_stock_prices_parquet_dir_path = "data/scraped/prices/*.parquet"
    merged_data_path                    = "data/processed/merged_raw_scraped_news_with_stock_prices.parquet"
    processed_merged_data_path          = "data/processed/processed_news_data.parquet"
    predictor_model_path                = "models/stock_price_predictor_forecaster.pkl"
    target_column                       = ColumnNames.CLOSE_PRICE_PCT_CHANGE.value

    train   = True
    predict = True

    # Instantiate
    spd = StockPricePredictor(predictor_path=predictor_model_path, use_gpu=False)

    # ===== 1) TRAIN & BACKTEST =====
    if train:
        # a) Train pipeline (this also saves the model)
        results_df = spd.run_stock_price_predictive_pipeline(
            raw_scraped_news_parquet_dir_path=raw_scraped_news_parquet_dir_path,
            raw_scraped_stock_prices_parquet_dir_path=raw_scraped_stock_prices_parquet_dir_path,
            merged_data_path=merged_data_path,
            processed_merged_data_path=processed_merged_data_path,
            predictor_model_path=predictor_model_path,
            use_processed_data=True,
            use_pretrained_model=False,
            merge_and_process_raw_scraped_data=False,
            target_column=target_column
        )

        # b) Reload the processed DataFrame & news‐only DF
        processed_df = pd.read_parquet(processed_merged_data_path)
        processed_df['stock_price_date'] = pd.to_datetime(processed_df['stock_price_date'])
        processed_df.set_index('stock_price_date', inplace=True)
        processed_df.sort_index(inplace=True)

        # only the numeric news feature(s) you used when training
        news_df = processed_df[['news_weight_on_stock_price_change']]

        # c) Run expanding‐window backtest (first 60 days to get started)
        bt = spd.backtest(
            df=processed_df,
            target_column=target_column,
            news_df=news_df,
            initial_train_size=60
        )

        # d) Compute metrics over the backtest
        metrics = spd.evaluate_backtest(bt)
        logging.info("=== Backtest Metrics ===")
        for k, v in metrics.items():
            logging.info(f"{k.upper():<6}: {v:.4f}")

    # ===== 2) ONE‐DAY FORECAST FROM TODAY’S NEWS =====
    if predict:
        # load the very last row of your processed news data
        daily = pd.read_parquet(processed_merged_data_path)
        daily['stock_price_date'] = pd.to_datetime(daily['stock_price_date'])
        daily.set_index('stock_price_date', inplace=True)
        # keep only the news feature(s)
        daily = daily[['news_weight_on_stock_price_change']].tail(1)

        # get next-day pct change
        next_pct = spd.predict_next_day_pct_change_from_news(
            daily_news_df=daily,
            predictor_model_path=predictor_model_path,
            news_cols=['news_weight_on_stock_price_change'],
            use_gpu=False
        )
        logging.info(f"Predicted next-day pct change: {next_pct * 100:.2f}%")