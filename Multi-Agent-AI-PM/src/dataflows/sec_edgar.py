import os
import warnings
import logging
import pandas as pd
from typing import Annotated

# Suppress noisy edgartools warnings about old filings lacking XBRL
warnings.filterwarnings("ignore", message=".*No XBRL attachments found.*")
warnings.filterwarnings("ignore", message=".*SGML fetch failed.*")
warnings.filterwarnings("ignore", message=".*Subheader.*not found in header.*")
logging.getLogger("edgar.core").setLevel(logging.ERROR)

# Ensure edgartools is available
try:
    from edgar import Company, set_identity
except ImportError as e:
    raise ImportError("edgartools is required for SEC EDGAR data. Install: pip install edgartools") from e


def _ensure_identity():
    """SEC requires an identity (name + email) for rate-limiting."""
    email = os.getenv("SEC_IDENTITY_EMAIL")
    if not email:
        email = "user@example.com"
    try:
        set_identity(email)
    except Exception:
        pass  # Already set or failed silently


def _extract_statement(
    ticker: str,
    statement_type: str,
    freq: str = "annual",
    curr_date: str | None = None,
) -> pd.DataFrame:
    """Extract and stitch financial statements from SEC EDGAR filings.

    Args:
        ticker: Stock ticker symbol
        statement_type: 'income_statement', 'balance_sheet', or 'cash_flow_statement'
        freq: 'annual' (10-K) or 'quarterly' (10-Q)
        curr_date: If provided, filter out periods after this date (yyyy-mm-dd)

    Returns:
        DataFrame with metrics as rows and periods as columns
    """
    _ensure_identity()
    company = Company(ticker.upper())

    form = "10-K" if freq.lower() == "annual" else "10-Q"
    filings = company.get_filings(form=form, amendments=False)

    all_periods: dict[str, pd.Series] = {}

    for filing in filings:
        try:
            obj = filing.obj()
            if not hasattr(obj, "financials"):
                continue

            fin = obj.financials
            if statement_type == "income_statement":
                stmt = fin.income_statement()
            elif statement_type == "balance_sheet":
                stmt = fin.balance_sheet()
            elif statement_type == "cash_flow_statement":
                stmt = fin.cash_flow_statement()
            else:
                raise ValueError(f"Unknown statement_type: {statement_type}")

            df = stmt.to_dataframe()
            mask = (~df["abstract"]) & (~df["dimension"])
            clean = df[mask]

            # Find period columns by detecting date-like column names
            # Some statements use '2025-09-27 (FY)', others just '2025-09-27'
            date_cols = []
            for c in clean.columns:
                try:
                    # Try parsing the date part (before any space)
                    date_part = c.split(" ")[0]
                    pd.Timestamp(date_part)
                    # Exclude known non-date columns
                    if c not in ("label", "concept", "standard_concept", "level",
                                 "balance", "weight", "preferred_sign",
                                 "parent_concept", "parent_abstract_concept",
                                 "is_breakdown", "dimension_axis", "dimension_member",
                                 "dimension_member_label", "dimension_label"):
                        date_cols.append(c)
                except Exception:
                    continue

            for col in date_cols:
                if col in all_periods:
                    continue
                period_data = clean[["label", col]].copy()
                period_data = period_data.dropna(subset=[col])
                period_data = period_data.drop_duplicates(subset="label", keep="first")
                if period_data.empty:
                    continue
                series = period_data.set_index("label")[col]
                all_periods[col] = series

        except Exception:
            # Skip filings that fail to parse (older filings often lack XBRL)
            continue

    if not all_periods:
        return pd.DataFrame()

    result = pd.DataFrame(all_periods)
    result.index.name = "Metric"

    # Clean column names: '2025-09-27 (FY)' -> '2025-09-27'
    result.columns = [c.split(" ")[0] for c in result.columns]

    # Deduplicate columns — keep the first occurrence of each date
    result = result.loc[:, ~result.columns.duplicated()]

    # Sort columns chronologically
    result = result.reindex(sorted(result.columns), axis=1)

    # Filter by curr_date if provided
    if curr_date:
        try:
            cutoff = pd.Timestamp(curr_date)
            valid_cols = [c for c in result.columns if pd.Timestamp(c) <= cutoff]
            result = result[valid_cols]
        except Exception:
            pass

    return result


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date for backtest filtering (yyyy-mm-dd)"] = None,
):
    """Get income statement data from SEC EDGAR filings.
    Provides 10-20+ years of historical data depending on filing availability."""
    df = _extract_statement(ticker, "income_statement", freq=freq, curr_date=curr_date)
    if df.empty:
        return f"No income statement data found for symbol '{ticker}'"
    return df.to_csv()


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date for backtest filtering (yyyy-mm-dd)"] = None,
):
    """Get balance sheet data from SEC EDGAR filings.
    Provides 10-20+ years of historical data depending on filing availability."""
    df = _extract_statement(ticker, "balance_sheet", freq=freq, curr_date=curr_date)
    if df.empty:
        return f"No balance sheet data found for symbol '{ticker}'"
    return df.to_csv()


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date for backtest filtering (yyyy-mm-dd)"] = None,
):
    """Get cash flow statement data from SEC EDGAR filings.
    Provides 10-20+ years of historical data depending on filing availability."""
    df = _extract_statement(ticker, "cash_flow_statement", freq=freq, curr_date=curr_date)
    if df.empty:
        return f"No cash flow data found for symbol '{ticker}'"
    return df.to_csv()


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for SEC EDGAR)"] = None,
):
    """Get latest company profile metrics from SEC EDGAR filings.
    Returns key metrics from the most recent 10-K filing."""
    _ensure_identity()
    company = Company(ticker.upper())

    try:
        filings = company.get_filings(form="10-K", amendments=False)
        if not filings:
            return f"No SEC filings found for symbol '{ticker}'"

        obj = filings[0].obj()
        if not hasattr(obj, "financials"):
            return f"No financial data in latest filing for '{ticker}'"

        # Use income statement to derive some basic metrics
        inc = obj.financials.income_statement().to_dataframe()
        mask = (~inc["abstract"]) & (~inc["dimension"])
        clean = inc[mask]

        # Extract revenue and net income from latest period
        date_cols = []
        for c in clean.columns:
            try:
                date_part = c.split(" ")[0]
                pd.Timestamp(date_part)
                if c not in ("label", "concept", "standard_concept", "level",
                             "balance", "weight", "preferred_sign",
                             "parent_concept", "parent_abstract_concept",
                             "is_breakdown", "dimension_axis", "dimension_member",
                             "dimension_member_label", "dimension_label"):
                    date_cols.append(c)
            except Exception:
                continue

        if not date_cols:
            return f"No financial metrics found for '{ticker}'"

        latest_col = date_cols[0]
        latest_date = latest_col.split(" ")[0]

        # Build a simple profile from available metrics
        metrics = {}
        for _, row in clean.iterrows():
            label = str(row["label"]).lower().strip()
            val = row.get(latest_col)
            if val and not pd.isna(val):
                metrics[label] = val

        # Map common labels
        def find_metric(*keys):
            for k in keys:
                for label, val in metrics.items():
                    if k in label:
                        return val
            return None

        revenue = find_metric("net sales", "revenue", "total revenues", "sales")
        net_income = find_metric("net income", "net earnings", "net profit")
        gross_profit = find_metric("gross margin", "gross profit")
        operating_income = find_metric("operating income")

        fields = [
            ("Name", company.name),
            ("Latest Fiscal Period", latest_date),
            ("Revenue", revenue),
            ("Net Income", net_income),
            ("Gross Profit", gross_profit),
            ("Operating Income", operating_income),
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


def get_earnings_dates(
    ticker: Annotated[str, "ticker symbol of the company"],
):
    """Get filing dates (10-K and 10-Q) as proxy for earnings dates from SEC EDGAR."""
    _ensure_identity()
    company = Company(ticker.upper())

    try:
        filings_10k = company.get_filings(form="10-K", amendments=False)
        filings_10q = company.get_filings(form="10-Q", amendments=False)

        dates = []
        for f in filings_10k:
            dates.append({"date": str(f.filing_date), "form": "10-K"})
        for f in filings_10q:
            dates.append({"date": str(f.filing_date), "form": "10-Q"})

        if not dates:
            return f"No filing dates found for symbol '{ticker}'"

        df = pd.DataFrame(dates)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
        return df.to_csv(index=False)

    except Exception as e:
        return f"Error retrieving filing dates for {ticker}: {str(e)}"
