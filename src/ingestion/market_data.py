"""
FEPSIP Market Data Ingestion
Async pipeline for OHLCV, fundamentals, and macro data.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import logger


class MarketDataIngester:
    """Async market data ingestion from Yahoo Finance."""

    def __init__(
        self,
        tickers: list[str],
        lookback_days: int = 365,
        interval: str = "1d",
    ) -> None:
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.interval = interval
        self._cache: dict[str, pd.DataFrame] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _fetch_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval=self.interval, auto_adjust=True)
        if df.empty:
            logger.warning("No data returned for ticker={}", ticker)
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df["ticker"] = ticker
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["rsi_14"] = self._compute_rsi(df["close"], 14)
        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    async def fetch_all(self) -> dict[str, pd.DataFrame]:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._fetch_ticker, ticker, start, end)
            for ticker in self.tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data: dict[str, pd.DataFrame] = {}
        for ticker, result in zip(self.tickers, results):
            if isinstance(result, Exception):
                logger.error("Failed to fetch {}: {}", ticker, result)
            elif isinstance(result, pd.DataFrame) and not result.empty:
                self._cache[ticker] = result
                data[ticker] = result
                logger.info("Fetched {} rows for {}", len(result), ticker)

        return data

    def get_combined_df(self) -> pd.DataFrame:
        if not self._cache:
            raise RuntimeError("No data loaded. Call fetch_all() first.")
        return pd.concat(self._cache.values())

    def get_returns_matrix(self) -> pd.DataFrame:
        dfs = []
        for ticker, df in self._cache.items():
            dfs.append(df[["returns"]].rename(columns={"returns": ticker}))
        return pd.concat(dfs, axis=1).dropna(how="all")


class FundamentalsIngester:
    """Fetch company fundamental data from yfinance."""

    KEYS = [
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "debtToEquity", "returnOnEquity", "revenueGrowth",
        "grossMargins", "ebitdaMargins", "totalRevenue", "ebitda",
        "totalDebt", "currentRatio", "quickRatio",
    ]

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _fetch_fundamentals(self, ticker: str) -> dict:
        info = yf.Ticker(ticker).info
        return {k: info.get(k) for k in self.KEYS} | {"ticker": ticker}

    async def fetch_all(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self._fetch_fundamentals, t)
            for t in self.tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rows = []
        for ticker, res in zip(self.tickers, results):
            if isinstance(res, Exception):
                logger.warning("Fundamentals fetch failed for {}: {}", ticker, res)
            else:
                rows.append(res)

        df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
        logger.info("Fetched fundamentals for {} tickers", len(df))
        return df


class MacroDataIngester:
    """Fetch macroeconomic indicators via yfinance proxy tickers."""

    MACRO_TICKERS = {
        "treasury_10y": "^TNX",
        "treasury_2y": "^IRX",
        "vix": "^VIX",
        "sp500": "^GSPC",
        "dxy": "DX-Y.NYB",
        "gold": "GC=F",
        "crude_oil": "CL=F",
    }

    def __init__(self, lookback_days: int = 365) -> None:
        self.lookback_days = lookback_days

    async def fetch_all(self) -> pd.DataFrame:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        results = {}
        for name, sym in self.MACRO_TICKERS.items():
            try:
                df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
                if not df.empty:
                    results[name] = df["Close"].squeeze()
            except Exception as e:
                logger.warning("Macro fetch failed for {}: {}", name, e)

        result_df = pd.DataFrame(results).dropna(how="all")
        logger.info("Fetched macro data: {} indicators", len(result_df.columns))
        return result_df


class NewsIngester:
    """Fetch financial news from Yahoo Finance."""

    def __init__(self, tickers: list[str], max_per_ticker: int = 20) -> None:
        self.tickers = tickers
        self.max_per_ticker = max_per_ticker

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _fetch_news(self, ticker: str) -> list[dict]:
        t = yf.Ticker(ticker)
        news = t.news or []
        articles = []
        for item in news[: self.max_per_ticker]:
            articles.append({
                "ticker": ticker,
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "timestamp": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                "summary": item.get("summary", ""),
            })
        return articles

    async def fetch_all(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._fetch_news, t) for t in self.tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for ticker, result in zip(self.tickers, results):
            if isinstance(result, Exception):
                logger.warning("News fetch failed for {}: {}", ticker, result)
            else:
                all_articles.extend(result)

        df = pd.DataFrame(all_articles) if all_articles else pd.DataFrame()
        logger.info("Fetched {} news articles", len(df))
        return df


class DataPipeline:
    """Orchestrates all data ingestion modules."""

    def __init__(self, tickers: list[str], lookback_days: int = 365) -> None:
        self.tickers = tickers
        self.market = MarketDataIngester(tickers, lookback_days)
        self.fundamentals = FundamentalsIngester(tickers)
        self.macro = MacroDataIngester(lookback_days)
        self.news = NewsIngester(tickers)

    async def run_all(self) -> dict[str, object]:
        logger.info("Starting full data pipeline for {} tickers...", len(self.tickers))
        market_data, fundamentals, macro_data, news_data = await asyncio.gather(
            self.market.fetch_all(),
            self.fundamentals.fetch_all(),
            self.macro.fetch_all(),
            self.news.fetch_all(),
        )
        logger.info("Data pipeline complete.")
        return {
            "market": market_data,
            "fundamentals": fundamentals,
            "macro": macro_data,
            "news": news_data,
        }
