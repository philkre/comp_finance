import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np


def pdr_download_prices_stooq(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Download daily OHLCV from Stooq using pandas_datareader.
    For US stocks, stooq often uses the format "APPL.US"
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    candidates = [
        ticker,
        ticker.upper(),
        f"{ticker}.US",
        f"{ticker.upper()}.US",
        f"{ticker}.us",
        f"{ticker.upper()}.us",
    ]

    last_err = None
    for t in candidates:
        try:
            df = pdr.DataReader(t, "stooq", start, end)
            if df is not None and not df.empty:
                # Stooq returns newest->oldest; sort to oldest->newest
                df = df.sort_index()
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to download data for ticker '{ticker}' from Stooq."
        f" Tried candidates: {candidates}. Last error: {last_err}"
    )


def download_prices_stooq(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Use yfinance to download daily OHLCV from Stooq.
    PDR is not supported for > python 3.11"""

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    candidates = [
        ticker,
        ticker.upper(),
        f"{ticker}.US",
        f"{ticker.upper()}.US",
        f"{ticker}.us",
        f"{ticker.upper()}.us",
    ]

    last_err = None
    for t in candidates:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to download data for ticker '{ticker}' from Stooq using yfinance."
        f" Tried candidates: {candidates}. Last error: {last_err}"
    )


def plot_stockprice(stock: pd.Series, ticker: str) -> None:
    """Plot the stock price over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(stock.index, stock.values, label=ticker)
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def annualised_mean_return(stock: pd.Series) -> float:
    """Annualised mean return for a given stock series assuming 252 trading days"""

    N = len(stock)
    assert N > 1, "Stock series must have more than one data point for mean estimation."

    returns = stock.pct_change().dropna()
    mean_return = returns["ASML"].mean() * 252
    return mean_return


def annualised_log_returns(stock: pd.Series) -> float:
    """Annualised log returns for a given stock series assuming 252 trading days"""

    N = len(stock)
    assert (
        N > 1
    ), "Stock series must have more than one data point for log return estimation."

    log_returns = np.log(stock / stock.shift(1)).dropna()
    mean_log_return = log_returns["ASML"].mean() * 252
    return mean_log_return


if __name__ == "__main__":

    ticker = "ASML"
    start_date = "2010-01-01"
    end_date = dt.date.today().strftime("%Y-%m-%d")

    data = download_prices_stooq(ticker, start_date, end_date)

    stock = data["Close"].astype(float)

    # plot_stockprice(stock, ticker)

    drift = annualised_mean_return(stock)
    log_drift = annualised_log_returns(stock)
    print(f"Annualised mean return for {ticker}: {drift}")
    print(f"Annualised log return for {ticker}: {log_drift}")
