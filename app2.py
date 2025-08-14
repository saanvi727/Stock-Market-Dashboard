import pandas as pd
import requests
import datetime
from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)
ALPHA_VANTAGE_API_KEY = "ZUTTFU8NZ3HCPBVW"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/compare")
def compare():
    return render_template("compare.html")

def fetch_daily_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" not in data:
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume"
    }, inplace=True)
    return df

@app.route("/data")
def data():
    symbol = request.args.get("symbol", "AAPL").upper()
    df = fetch_daily_data(symbol)
    if df is None or df.empty:
        return {"error": f"Invalid symbol or no data found for {symbol}."}

    df = df.last("365D").copy()

    df["30d_MA"] = df["Close"].rolling(window=30).mean()
    df["90d_MA"] = df["Close"].rolling(window=90).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["BB_MA"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["BB_MA"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["BB_MA"] - 2 * df["Close"].rolling(window=20).std()

    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["Cumulative_TPV"] = (df["Typical_Price"] * df["Volume"]).cumsum()
    df["Cumulative_Volume"] = df["Volume"].cumsum()
    df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]

    df.dropna(subset=["Close", "Volume", "30d_MA", "90d_MA", "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower", "VWAP"], inplace=True)

    return {
        "symbol": symbol,
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "closes": df["Close"].tolist(),
        "volumes": df["Volume"].tolist(),
        "ma30": df["30d_MA"].tolist(),
        "ma90": df["90d_MA"].tolist(),
        "rsi": df["RSI"].tolist(),
        "macd": df["MACD"].tolist(),
        "macd_signal": df["MACD_Signal"].tolist(),
        "bb_upper": df["BB_Upper"].tolist(),
        "bb_lower": df["BB_Lower"].tolist(),
        "vwap": df["VWAP"].tolist(),
        "high": df["High"].max(),
        "low": df["Low"].min(),
        "last_close": df["Close"].iloc[-1]
    }

@app.route("/compare-data")
def compare_data():
    symbols = request.args.get("symbols", "")
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    if not symbol_list:
        return jsonify({"error": "No valid symbols provided."})

    results = {}

    for symbol in symbol_list:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "Time Series (Daily)" not in data:
            results[symbol] = {"error": f"No data found for {symbol}"}
            continue

        ts = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume"
        })

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.tail(365)  # Keep last 1 year of data

        df = df.astype(float)
        df["30d_MA"] = df["Close"].rolling(window=30).mean()
        df["90d_MA"] = df["Close"].rolling(window=90).mean()
        df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["Cumulative_TPV"] = (df["Typical_Price"] * df["Volume"]).cumsum()
        df["Cumulative_Volume"] = df["Volume"].cumsum()
        df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]

        df.dropna(subset=["Close", "30d_MA", "90d_MA", "VWAP"], inplace=True)

        df["Normalized_Close"] = (df["Close"] / df["Close"].iloc[0]) * 100
        df["Normalized_MA30"] = (df["30d_MA"] / df["Close"].iloc[0]) * 100
        df["Normalized_MA90"] = (df["90d_MA"] / df["Close"].iloc[0]) * 100
        df["Normalized_VWAP"] = (df["VWAP"] / df["Close"].iloc[0]) * 100

        results[symbol] = {
            "dates": df.index.strftime("%Y-%m-%d").tolist(),
            "close": df["Normalized_Close"].tolist(),
            "ma30": df["Normalized_MA30"].tolist(),
            "ma90": df["Normalized_MA90"].tolist(),
            "vwap": df["Normalized_VWAP"].tolist()
        }

        time.sleep(15)  # Respect Alpha Vantage free-tier rate limit (5 requests/min)

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)