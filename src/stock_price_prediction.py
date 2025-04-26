import os
from tqdm import tqdm
from torch import device, cuda
import pickle
import logging
import pandas as pd
import polars as pl
import numpy as np
import glob
from enum import Enum
import spacy
from multiprocessing import cpu_count
from typing import Any, Tuple, Union
from gensim import models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
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
    def __init__(self, predictor_path: str):
        self.use_gpu = False
        self.device = device('cuda' if cuda.is_available() else 'cpu') if self.use_gpu else device('cpu') # Use GPU if available.
        self.predictor_model_path = predictor_path  # Path to the saved model.

    def create_features(self, df: pd.DataFrame, label: str = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Create features from the given dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        label : str, optional
            Name of the column to use as the target, by default None.

        Returns
        -------
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
            If `label` is not `None`, returns a tuple of the feature dataframe and the target series.
            Otherwise, returns only the feature dataframe.
        """
        
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        
        X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth']]
        if label:
            y = df[label]
            return X, y
        return X
    
    def split_data(self, df: pd.DataFrame, days_to_predict: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the given DataFrame into a training set and a test set.

        The test set will contain the last `days_to_predict` days of the DataFrame.
        The training set will contain all the other data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be split.
        days_to_predict : int, optional
            Number of days to predict. Defaults to 1.

        Returns
        -------
        train_data : pd.DataFrame
            Training set.
        test_data : pd.DataFrame
            Test set.
        """
        
        # TODO: The test data must contain the time I want to train my model to predict.
        # If I want my model to predict 3 days, the test data must contain the last 3 days.
        # I can use cross validation by setting any last 3 days as test data. The model won't know.
        # For example, if I have a dataframe from 1-1-2020 to 28-2-2025, I can use 28-2-2025 as test data,
        # or keep a dataframe from 1-1-2020 to 18-2-2025 and use 19-2-2025 as test data.
        # NOTE: It won't make sense to predict more than 1 day ahead, since my purpose is to prepare my self for the next day.
        df.index = pd.to_datetime(df.index)
        last_date = df.index.max()

        # Calculate the start dates for test and validation sets
        test_start_date = last_date - pd.DateOffset(days=days_to_predict)

        # Split the DataFrame based on dates
        test_data = df.loc[df.index >= pd.Timestamp(test_start_date)].copy()
        train_data = df.loc[df.index < pd.Timestamp(test_start_date)].copy()
                        
        return train_data, test_data
    
    def train_model(self, df: pd.DataFrame, target_column: str) -> XGBRegressor:
        """
        Trains a model on the given DataFrame and target column.

        The model is a XGBRegressor and is trained using GridSearchCV with TimeSeriesSplit cross-validation.
        The model is trained on the given DataFrame after splitting it into a training set and a test set.
        The test set is used for hyperparameter tuning and the model is evaluated on the test set.
        The best model is then saved to the given filepath.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data to be used for training.
        target_column : str
            Name of the column in the DataFrame containing the target values.

        Returns
        -------
        forecaster : XGBRegressor
            The trained model.
        """
        # TODO: Train for only one stock, remove from dataframe the rest to avoid overfitting and having irrelevant features
        logging.info("Training model...")
        cv_split = TimeSeriesSplit(n_splits=4, test_size=100)
        parameters = {
            "max_depth": [3, 4, 6, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [100, 300, 500, 700, 900, 1000],
            "colsample_bytree": [0.3, 0.5, 0.7]
        }
        train_data, test_data = self.split_data(df=df)
        X_train, y_train = self.create_features(train_data, label=target_column)
        X_test, y_test = self.create_features(test_data, label=target_column)
        reg = XGBRegressor()
        grid_search = GridSearchCV(estimator=reg, cv=cv_split, param_grid=parameters)
        forecaster = grid_search.fit(X_train, y_train, verbose=False) # Change verbose to True if you want to see it train
        logging.info("Model trained.")
        self.save_model(filepath=self.predictor_model_path, model=forecaster)
        logging.info(f"Model saved to {self.predictor_model_path}")
        return forecaster

    def evaluate_model(self, test_data: pd.DataFrame, target_column: str) -> None:
        """
        Evaluate the performance of the model on the test set.

        Parameters
        ----------
        test_data : pd.DataFrame
            DataFrame containing the test data.
        target_column : str
            Name of the column containing the target time series.
        prediction_column : str, default='Prediction'
            Name of the column containing the predicted time series.

        Returns
        -------
        None

        Notes
        -----
        The metrics used are Mean Squared Error (MSE), R-Squared (R2), Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
        """
        logging.info("Evaluating model...")
        mse = mean_squared_error(test_data[target_column], test_data[ColumnNames.PRICE_CHANGE_PREDICTION.value])
        r2 = r2_score(test_data[target_column], test_data[ColumnNames.PRICE_CHANGE_PREDICTION.value])
        mae = mean_absolute_error(test_data[target_column], test_data[ColumnNames.PRICE_CHANGE_PREDICTION.value])
        mape = mean_absolute_percentage_error(test_data[target_column], test_data[ColumnNames.PRICE_CHANGE_PREDICTION.value])
        logging.info(f'Test MSE: {mse:.2f}')
        logging.info(f'Test R2: {r2:.2f}')
        logging.info(f'Test MAE: {mae:.2f}')
        logging.info(f'Test MAPE: {mape:.2f}')

    def predict_price_change(self, model: Any, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Makes predictions on the test set using the given model and DataFrame.
        Then, combines the test and train data into a single DataFrame, and returns it.
        The combined DataFrame will have a 'Prediction' column containing the predicted
        stock prices.
        
        Parameters
        ----------
        model : Any
            The model to use for prediction.
        df : pd.DataFrame
            DataFrame containing the data to be used for prediction.
        target_column : str
            Name of the column containing the target time series.
        
        Returns
        -------
        df_all : pd.DataFrame
            Combined DataFrame containing the train and test data.
        train_data : pd.DataFrame
            DataFrame containing the train data.
        test_data : pd.DataFrame
            DataFrame containing the test data.
        """
        train_data, test_data = self.split_data(df=df)
        X_train, y_train = self.create_features(train_data, label=target_column)
        X_test, y_test = self.create_features(test_data, label=target_column)
        test_data[ColumnNames.PRICE_CHANGE_PREDICTION.value] = model.predict(X_test)
        df_all = pd.concat([test_data, train_data], sort=False)
        return df_all, train_data, test_data

    def save_model(self, filepath: str, model: Any) -> None:
        """
        Saves the given model to a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to which to save the model.
        model : Any
            The model to be saved.

        Returns
        -------
        None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f'Model saved to {filepath}')

    def load_model(self, filepath: str) -> XGBRegressor:
        """
        Loads the model from the given file path using pickle.

        Parameters
        ----------
        filepath : str
            The path from which to load the model.

        Returns
        -------
        XGBRegressor
            The loaded model.
        """
        with open(filepath, 'rb') as f:
            forecaster = pickle.load(f)
        logging.info(f'Model loaded from {filepath}')
        return forecaster


def run_stock_price_predictive_pipeline():
    raw_scraped_news_parquet_dir_path = "data/scraped/news/*.parquet"
    raw_scraped_stock_prices_parquet_dir_path = "data/scraped/prices/*.parquet"
    predictor_model_path = 'models/stock_price_predictor_forecaster.pkl'
    merged_data_path = 'data/processed/merged_raw_scraped_news_with_stock_prices.parquet'
    processed_merged_data_path = "data/processed/processed_news_data.parquet"
    use_processed_data = True
    use_pretrained_model = True
    merge_and_process_raw_scraped_data = False
    target_column = ColumnNames.CLOSE_PRICE_PCT_CHANGE.value
    
    pnd = PreprocessNewsData()
    cnd = ClassifyNewsData()
    spp = StockPricePredictor(predictor_path=predictor_model_path)
    
    # -------- Process Raw Scraped Data --------
    if merge_and_process_raw_scraped_data:
        # News
        merged_scraped_news_export_path = pnd.merge_parquet_data(directory=raw_scraped_news_parquet_dir_path, data_category=ScrapedDataCategory.NEWS.value)
        preprocess_scraped_news_data_df = pnd.preprocess_scraped_news_data(filepath=merged_scraped_news_export_path)
        # Stock Prices
        merged_scraped_stock_prices_export_path = pnd.merge_parquet_data(directory=raw_scraped_stock_prices_parquet_dir_path, data_category=ScrapedDataCategory.PRICES.value)
        preprocess_scraped_stock_prices_data_df = pnd.preprocess_scraped_stock_prices_data(filepath=merged_scraped_stock_prices_export_path)
        
        merged_data = pnd.merge_scraped_news_with_prices_data(news_df=preprocess_scraped_news_data_df, stock_prices_df=preprocess_scraped_stock_prices_data_df, export_path=merged_data_path)
    else:
        merged_data = pnd.load_news_df(data_path=merged_data_path)
        merged_data.index = pd.to_datetime(merged_data.index)
        
    if not use_processed_data:
        merged_data = cnd.preprocess_df(df=merged_data)
        
        all_news_df, daily_news_df = spp.split_data(df=merged_data)
    else:
        all_news_df = pd.read_parquet(processed_merged_data_path)
        
        all_news_df, daily_news_df = spp.split_data(df=all_news_df)
            
    if target_column not in all_news_df.columns:
        raise ValueError(f"Target column {target_column} not found in dataframe columns.")
    
    # ------- Machine Learning Pipeline -------
    if not use_pretrained_model:
        forecaster = spp.train_model(df=all_news_df, target_column=target_column)
    else:
        with open(predictor_model_path, 'rb') as f:
            forecaster = pickle.load(f)
    
    df_with_predictions, train_data, test_data = spp.predict_price_change(model=forecaster, df=daily_news_df, target_column=target_column)
    spp.evaluate_model(test_data=test_data, target_column=target_column)

    latest_date = df_with_predictions.index.max()
    latest_prediction = df_with_predictions.loc[latest_date][ColumnNames.PRICE_CHANGE_PREDICTION.value]
    latest_prediction_pct = latest_prediction * 100
    change_type = "positive" if latest_prediction_pct > 0 else "negative"
    logging.info(f"Latest prediction: {latest_prediction_pct:.2f}% with date {latest_date} ({change_type} change)")
            
            
if __name__ == "__main__":
    run_stock_price_predictive_pipeline()