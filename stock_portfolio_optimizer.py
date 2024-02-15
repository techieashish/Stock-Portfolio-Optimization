from portfolio_optimizers import PortfolioOptimizer
from eda_sharpe_plotter import EDA_SHARPE_Optimizer
from markowtiz_optimizer import MarkPortfolioOptimizer
from datetime import datetime, timedelta
import os
import pandas as pd
import warnings


warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning)

basepath = os.path.abspath(os.path.dirname(__file__))
csv_file_paths = os.path.join(basepath, 'nifty_50/')


def main():
    stock_data = load_data_from_csv()
    analyse_portfolio(stock_data)
    plot_eda_sharpe(stock_data)
    analyse_using_markowtiz(stock_data)


def load_data_from_csv():
    combined_stocks_data = dict()
    for csv_file in os.listdir(csv_file_paths):
        data_dict = {}
        df = pd.read_csv(os.path.join(basepath, csv_file_paths + '/' + csv_file))
        end_date = datetime.strptime('2023-11-21', '%Y-%m-%d')
        start_date = end_date - timedelta(days=90)
        df_filtered = df[df['Date'].between(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]
        for index, row in df.iterrows():
            date = row['Date']
            data_dict[date] = row.drop('Date').tolist()
        combined_stocks_data[csv_file.replace('.csv', '')] = data_dict
    return combined_stocks_data


def analyse_portfolio(stock_data):
    optimizer = PortfolioOptimizer(stock_data)
    optimal_weights = optimizer.optimize_portfolio()
    print("Optimal Portfolio Weights:", optimal_weights)


def plot_eda_sharpe(stock_data):
    optimizer = EDA_SHARPE_Optimizer(stock_data)
    optimizer.plot_eda()
    optimizer.plot_sharpe_ratios()


def analyse_using_markowtiz(stock_data):
    optimizer = MarkPortfolioOptimizer(stock_data)
    optimal_weights = optimizer.minimize_volatility()
    optimizer.print_optimized_weights(optimal_weights)
    optimizer.write_to_excel(optimal_weights)


if __name__ == '__main__':
    main()

