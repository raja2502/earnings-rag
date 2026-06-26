import os
import re
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

# --- Companies we're tracking ---
COMPANIES = {
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "BLK": "BlackRock",
    "RITM": "Rithm Capital",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AAPL": "Apple",
    "NVDA": "NVIDIA"
}

# --- Quarters we're covering ---
QUARTERS = [
    ("Q1", "2025"), ("Q2", "2025"),
    ("Q3", "2025"), ("Q4", "2025"),
    ("Q1", "2026")
]

TRANSCRIPT_DIR = Path("data/transcripts")
FINANCIAL_DIR = Path("data/financials")


def clean_text(text: str) -> str:
    """Remove extra whitespace, special characters, and empty lines."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def fetch_motley_fool_transcript(ticker: str, company: str, quarter: str, year: str) -> str:
    """
    Search and fetch earnings call transcript from Motley Fool.
    Falls back to a placeholder if not found.
    """
    search_url  = (
        f"https://tickertrends.io/transcripts/{ticker}/{quarter}-earnings-transcript-{year}"
    )
    # search_url = (
    #     f"https://www.fool.com/search/solr.aspx?q="
    #     f"{ticker}+{quarter}+{year}+earnings+call+transcript"
    # )
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        page_resp = requests.get(search_url, headers=headers, timeout=10)
        page_resp.raise_for_status()

        page_soup = BeautifulSoup(page_resp.text, "html.parser")

        # Extract transcript title
        title_tag = page_soup.find("h1")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

        # Extract transcript container
        transcript_div = page_soup.find(
            "div",
            class_="mt-6 h-[85vh] overflow-auto rounded-lg border-2 pl-2 text-lg text-gray-500"
        )

        if not transcript_div:
            print(f"  Warning: Transcript div not found for {ticker} {quarter} {year}")
            return ""

        paragraphs = []

        for p in transcript_div.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

        if not paragraphs:
            print(f"  Warning: No transcript text found for {ticker} {quarter} {year}")
            return ""

        transcript_text = "\n\n".join(paragraphs)

        final_text = f"{title}\n\n{transcript_text}" if title else transcript_text

        return final_text.strip()

    except requests.exceptions.HTTPError as e:
        print(f"  Warning: Transcript page not found for {ticker} {quarter} {year} — {e}")

    except Exception as e:
        print(f"  Warning: Could not fetch {ticker} {quarter} {year} — {e}")

    return ""


def save_transcript(ticker: str, quarter: str, year: str, text: str):
    """Save transcript to file."""
    filename = TRANSCRIPT_DIR / f"{ticker}_{quarter}_{year}.txt"
    with open(filename, "w") as f:
        f.write(text)
    print(f"  Saved: {filename}")


def fetch_all_transcripts():
    """Fetch transcripts for all companies and quarters."""
    print("\nFetching transcripts...")
    for ticker, company in COMPANIES.items():
        for quarter, year in QUARTERS:
            filename = TRANSCRIPT_DIR / f"{ticker}_{quarter}_{year}.txt"
            if filename.exists():
                print(f"  Already exists: {ticker} {quarter} {year}")
                continue
            print(f"  Fetching: {ticker} {quarter} {year}")
            text = fetch_motley_fool_transcript(ticker, company, quarter, year)
            if text:
                save_transcript(ticker, quarter, year, text)
            else:
                print(f"  No transcript found for {ticker} {quarter} {year}")


# def fetch_financials_from_macrotrends(ticker: str, company: str):
#     """
#     Fetch income statement data from Macrotrends for a company.
#     Saves as CSV in data/financials/.
#     """
#     filename = FINANCIAL_DIR / f"{ticker}_financials.csv"
#     if filename.exists():
#         print(f"  Already exists: {ticker} financials")
#         return

#     # Macrotrends income statement URL pattern
#     company_slug = company.lower().replace(" ", "-")
#     url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company_slug}/income-statement"
#     headers = {"User-Agent": "Mozilla/5.0"}

#     try:
#         resp = requests.get(url, headers=headers, timeout=10)
#         soup = BeautifulSoup(resp.text, "html.parser")

#         # Find the first data table
#         table = soup.find("table")
#         if table:
#             df = pd.read_html(str(table))[0]
#             df.to_csv(filename, index=False)
#             print(f"  Saved: {filename}")
#         else:
#             print(f"  No financial table found for {ticker}")

#     except Exception as e:
#         print(f"  Warning: Could not fetch financials for {ticker} — {e}")

def clean_financial_value(value: str):
    value = value.strip()

    if value in ["", "-"]:
        return None

    value = value.replace("$", "").replace(",", "")

    try:
        return float(value)
    except ValueError:
        return value


def fetch_financials_from_macrotrends(ticker: str, company: str):
    """
    Fetch income statement data from Macrotrends using Selenium.
    Saves as CSV in data/financials/.
    """

    ticker = ticker.upper().strip()
    company_slug = company.lower().replace(" ", "-")

    filename = FINANCIAL_DIR / f"{ticker}_financials.csv"

    if filename.exists():
        print(f"  Already exists: {ticker} financials")
        return

    url = (
        f"https://www.macrotrends.net/stocks/charts/"
        f"{ticker}/{company_slug}/income-statement"
    )

    options = Options()
    options.page_load_strategy = "eager"   # important: don't wait for all ads/scripts

    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--blink-settings=imagesEnabled=false")

    driver = webdriver.Chrome(options=options)

    try:
        driver.set_page_load_timeout(30)
        driver.get(url)

        # Wait only for the grid, not the full page
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "jqxgrid"))
        )

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        grid = soup.find("div", id="jqxgrid")

        if not grid:
            print(f"  No jqxgrid found for {ticker}")
            return

        date_columns = []

        for col in grid.find_all(attrs={"role": "columnheader"}):
            text = col.get_text(" ", strip=True)
            matches = re.findall(r"\d{4}-\d{2}-\d{2}", text)
            date_columns.extend(matches)

        date_columns = list(dict.fromkeys(date_columns))

        if not date_columns:
            print(f"  No date columns found for {ticker}")
            return

        rows = []

        for row in grid.find_all(attrs={"role": "row"}):
            cells = row.find_all(attrs={"role": "gridcell"})

            if not cells:
                continue

            metric = cells[0].get_text(" ", strip=True)

            if not metric:
                continue

            # cells[0] = metric
            # cells[1] = chart icon
            # cells[2:] = values
            value_cells = cells[2:]

            values = [
                clean_financial_value(cell.get_text(" ", strip=True))
                for cell in value_cells
            ]

            values = values[:len(date_columns)]

            row_dict = {"metric": metric}

            for date, value in zip(date_columns, values):
                row_dict[date] = value

            rows.append(row_dict)

        if not rows:
            print(f"  No financial rows found for {ticker}")
            return

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        print(f"  Saved: {filename}")

    except Exception as e:
        print(f"  Warning: Could not fetch financials for {ticker} — {e}")

    finally:
        driver.quit()

def fetch_all_financials():
    """Fetch financials for all companies."""
    print("\nFetching financials...")
    for ticker, company in COMPANIES.items():
        print(f"  Fetching: {ticker}")
        fetch_financials_from_macrotrends(ticker, company)


if __name__ == "__main__":
    print("Starting data ingestion...")
    fetch_all_transcripts()
    fetch_all_financials()
    print("\nDone. Check data/transcripts/ and data/financials/")