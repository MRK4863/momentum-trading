# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a momentum trading analysis project that tracks stock performance using yfinance data. The project analyzes Indian stocks (NSE) to identify momentum patterns over 6-day and 30-day periods.

## Development Commands

Since this is a Python-based data analysis project, use these commands:

- Run the main analysis script: `python roughWork.py`
- Launch Jupyter notebook for interactive analysis: `jupyter notebook roughWork.ipynb`
- Install dependencies: `pip install yfinance pandas datetime`

## Architecture

### Core Components

1. **Data Sources**:
   - `METADATA.csv` / `instruments_data.csv`: Contains stock instrument metadata with categorization
   - Yahoo Finance API (via yfinance): Real-time stock price data

2. **Main Analysis Function** (`yt_finance_historical_data`):
   - Downloads 40 days of historical data for Indian stocks (NSE format with .NS suffix)
   - Calculates momentum over 6-day and 30-day periods
   - Computes price differences and percentage changes
   - Handles missing data using forward-fill
   - Sorts results by momentum performance

3. **Data Processing Flow**:
   - Load instrument list from CSV metadata
   - Append `.NS` suffix for NSE symbols
   - Download historical price data
   - Calculate momentum metrics (last price vs 4 trading days prior)
   - Merge with metadata for enhanced analysis

### Key Data Structures

- **Stock Metadata**: Contains instrument symbols, market cap categories, personal ratings, and symbol categories
- **Price DataFrames**: 6-day and 30-day momentum analysis with percentage changes
- **Momentum Metrics**: Price differences and percentage changes sorted by performance

### Notes

- The project handles delisted/invalid stocks gracefully (shows warnings but continues)
- Uses deprecated pandas `fillna(method='ffill')` - consider updating to `.ffill()`
- Data is focused on Indian stock market (NSE) with `.NS` symbol format
- Momentum calculation: (current_price - price_4_days_ago) / price_4_days_ago