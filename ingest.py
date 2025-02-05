import argparse
import os

import yfinance as yf


def download_stock_data(input_file, output_directory):
    """Reads the input file, fetches stock data, and saves it to the output directory."""
    os.makedirs(output_directory, exist_ok=True)

    with open(input_file, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) != 3:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            ticker, start_date, end_date = parts
            print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    print(f"No data found for {ticker}. Skipping.")
                    continue

                output_file = os.path.join(output_directory, f"{ticker}.csv")
                data.columns = data.columns.droplevel(1)
                data.to_csv(output_file)
                print(f"Saved {ticker} data to {output_file}")

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Stock Data Ingestion Script")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing stock tickers and date ranges.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Directory where output CSV files will be saved (default: ./dataa).",
    )

    args = parser.parse_args()
    download_stock_data(args.input_file, args.output_directory)


if __name__ == "__main__":
    main()
