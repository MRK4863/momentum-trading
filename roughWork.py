import yfinance as yf
import datetime
import pandas as pd

def yt_finance_historical_data(symbols_list):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=40)  # enough buffer for 30 trading days
    df = yf.download(symbols_list, start=start, end=end, interval="1d")
    last_6 = df.tail(6)["Close"]
    last_30 = df.tail(40)["Close"]

    last_6, last_30 = last_6.T, last_30.T
    last_6 = last_6.reset_index()
    last_30 = last_30.reset_index()

    last_6['SYMBOL'] = last_6['Ticker'].str.replace('.NS', '')
    last_30['SYMBOL'] = last_30['Ticker'].str.replace('.NS', '')

    # Fill NaN values in date columns starting from the second date column
    date_cols_6 = last_6.columns[1:-1]  # exclude 'Ticker' and 'SYMBOL'
    date_cols_30 = last_30.columns[1:-1]  # exclude 'Ticker' and 'SYMBOL'
    last_6[date_cols_6[1:]] = last_6[date_cols_6[1:]].fillna(method='ffill', axis=1)
    last_30[date_cols_30[1:]] = last_30[date_cols_30[1:]].fillna(method='ffill', axis=1)

    # For last_6
    last_date_6 = date_cols_6[-1]
    fourth_from_last_6 = date_cols_6[-4]
    last_6['DIFF'] = last_6[last_date_6] - last_6[fourth_from_last_6]
    last_6['Diff_percent'] = last_6['DIFF'] *100 / last_6[fourth_from_last_6]

    # For last_30
    last_date_30 = date_cols_30[-1]
    fourth_from_last_30 = date_cols_30[-4]
    last_30['DIFF'] = last_30[last_date_30] - last_30[fourth_from_last_30]
    last_30['Diff_percent'] = last_30['DIFF'] *100 / last_30[fourth_from_last_30]

    # Sort by Diff_percent in descending order
    last_6 = last_6.sort_values(by='Diff_percent', ascending=False)
    last_30 = last_30.sort_values(by='Diff_percent', ascending=False)

    return last_6, last_30

df_metadata = pd.read_csv("METADATA.csv")
sample_list = df_metadata["Instrument"].to_list()
sample_list = [sample+".NS" for sample in sample_list]

last_6, last_30 = yt_finance_historical_data(sample_list)

merged_df = pd.merge(df_metadata, last_6, left_on='Instrument', right_on='SYMBOL', how="left")
merged_df = merged_df.sort_values(by='Diff_percent', ascending=False)