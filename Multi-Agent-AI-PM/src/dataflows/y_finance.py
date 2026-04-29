from typing import Annotated
from datetime import datetime
import yfinance as yf

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    return csv_string

def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get company fundamentals overview from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = ticker_obj.info

        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")),
            ("Sector", info.get("sector")),
            ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")),
            ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")),
            ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")),
            ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")),
            ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")),
            ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")),
            ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")),
            ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")),
            ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")),
            ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")),
            ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")),
            ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")),
            ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        import csv
        import io
        present = [(label, value) for label, value in fields if value is not None]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([label for label, _ in present])
        writer.writerow([value for _, value in present])
        return buf.getvalue()

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        return data.to_csv(index_label="Metric")
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"

        return data.to_csv(index_label="Metric")
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"

        return data.to_csv(index_label="Metric")
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_earnings_dates(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get earnings dates with actual and estimated EPS from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.earnings_dates

        if data is None or data.empty:
            return f"No earnings dates data found for symbol '{ticker}'"

        return data.to_csv()

    except Exception as e:
        return f"Error retrieving earnings dates for {ticker}: {str(e)}"


def get_quarterly_history(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get quarterly price history (OHLCV) from yfinance, all available data."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.history(period="max", interval="3mo")

        if data.empty:
            return f"No quarterly history data found for symbol '{ticker}'"

        # Remove timezone info from index for cleaner output
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Round numerical values to 2 decimal places for cleaner display
        numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].round(2)

        return data.to_csv()

    except Exception as e:
        return f"Error retrieving quarterly history for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"]
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"

        return data.to_csv()
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"


def screen_stocks(
    query=None,
    size: int = 100,
    sort_field: str = "ticker",
    sort_asc: bool = False,
):
    """Screen stocks using yfinance screener.

    Args:
        query: A predefined screener name (str) or a yf.EquityQuery object.
               Predefined names: 'aggressive_small_caps', 'day_gainers', 'day_losers',
               'growth_technology_stocks', 'most_actives', 'undervalued_growth_stocks'.
        size: Number of results (max 250).
        sort_field: Column to sort by.
        sort_asc: Sort ascending if True.

    Returns:
        Dict with 'tickers' (list[str]) and 'data' (list[dict]) of screened stocks.
    """
    import time

    kwargs = {"sortField": sort_field, "sortAsc": sort_asc}
    if isinstance(query, str):
        kwargs["query"] = query
        kwargs["count"] = min(size, 250)
    else:
        kwargs["query"] = query
        kwargs["size"] = min(size, 250)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = yf.screen(**kwargs)

            if not result or "quotes" not in result:
                return {"tickers": [], "data": []}

            quotes = result["quotes"]
            tickers = [q.get("symbol", "") for q in quotes if q.get("symbol")]
            return {"tickers": tickers, "data": quotes}

        except Exception as e:
            err_str = str(e)
            if "Rate" in err_str or "429" in err_str or "Too Many" in err_str:
                wait = 5 * (attempt + 1)
                print(f"Rate limited on screen attempt {attempt + 1}/{max_retries}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"Error screening stocks: {e}")
            return {"tickers": [], "data": []}

    print("Error screening stocks: rate limit retries exhausted")
    return {"tickers": [], "data": []}


def get_sector_top_companies(
    sector_key: Annotated[str, "sector key, e.g. 'technology', 'healthcare'"],
):
    """Get top companies in a sector from yfinance.

    Valid sector keys: basic-materials, communication-services, consumer-cyclical,
    consumer-defensive, energy, financial-services, healthcare, industrials,
    real-estate, technology, utilities.
    """
    try:
        sector = yf.Sector(sector_key)
        data = sector.top_companies

        if data is None or data.empty:
            return f"No top companies data for sector '{sector_key}'"

        return data.reset_index().to_dict(orient="records")

    except Exception as e:
        return f"Error getting sector top companies for {sector_key}: {str(e)}"


def get_industry_top_companies(
    industry_key: Annotated[str, "industry key, e.g. 'semiconductors'"],
):
    """Get top and top growth companies in an industry from yfinance."""
    try:
        industry = yf.Industry(industry_key)
        result = {}

        top = industry.top_companies
        if top is not None and not top.empty:
            result["top_companies"] = top.reset_index().to_dict(orient="records")

        growth = industry.top_growth_companies
        if growth is not None and not growth.empty:
            result["top_growth_companies"] = growth.reset_index().to_dict(orient="records")

        if not result:
            return f"No company data for industry '{industry_key}'"

        return result

    except Exception as e:
        return f"Error getting industry companies for {industry_key}: {str(e)}"

def get_analyst_recommendations(symbol: Annotated[str, "ticker symbol of the company"]):
    """Get analyst recommendations for a ticker from yfinance."""
    try:
        ticker = yf.Ticker(symbol.upper())
        recs = ticker.recommendations
        if recs is None or recs.empty:
            return f"No analyst recommendations found for {symbol}"
        return recs.reset_index().to_dict(orient="records")
    except Exception as e:
        return f"Error getting analyst recommendations for {symbol}: {str(e)}"


def get_growth_estimates(symbol: Annotated[str, "ticker symbol of the company"]):
    """Get growth estimates for a ticker from yfinance."""
    try:
        ticker = yf.Ticker(symbol.upper())
        growth = ticker.growth_estimates
        if growth is None or growth.empty:
            return f"No growth estimates found for {symbol}"
        return growth.reset_index().to_dict(orient="records")
    except Exception as e:
        return f"Error getting growth estimates for {symbol}: {str(e)}"
