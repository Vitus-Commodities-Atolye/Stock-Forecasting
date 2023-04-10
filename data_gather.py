import yfinance as yf
import pandas as pd

# Define the ticker symbol and period of interest
ticker = "AAPL"
period = "1mo"

# Fetch the data using yfinance
data = yf.download(ticker, period=period)

# Remove the current month's data
data = data[data.index.month != pd.Timestamp.now().month]

# Save the data as a CSV file
data.to_csv(f"{ticker}_{period}.csv", index=True)
