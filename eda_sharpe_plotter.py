import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class EDA_SHARPE_Optimizer(object):
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.returns = self.calculate_daily_returns()
        self.expected_returns, self.covariance_matrix = self.get_expected_returns_and_covariance()

    def calculate_daily_returns(self):
        returns = {}
        for stock, data in self.stock_data.items():
            prices = pd.Series([float(v[4]) if v[4] != '#VALUE!' else np.nan for v in data.values()], index=data.keys())
            daily_returns = prices.pct_change().fillna(0)
            returns[stock] = daily_returns
        return pd.DataFrame(returns)

    def get_expected_returns_and_covariance(self):
        expected_returns = self.returns.mean()
        covariance_matrix = self.returns.cov()
        return expected_returns, covariance_matrix

    def calculate_portfolio_performance(self, weights):
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def plot_eda(self):
        plt.figure(figsize=(14, 7))
        for stock in self.stock_data.keys():
            dates = list(self.stock_data[stock].keys())
            prices = [v[4] for v in self.stock_data[stock].values()]

            # Convert string dates to datetime objects with the correct format
            datetime_index = pd.to_datetime(dates, format='%d-%m-%Y')

            plt.plot(datetime_index, prices, label=stock)
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.title('Stock Adjusted Close Price Over Time')
        plt.legend()
        plt.show()

    def plot_sharpe_ratios(self):
        num_portfolios = 10000
        all_weights = np.zeros((num_portfolios, len(self.stock_data.keys())))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for ind in range(num_portfolios):
            weights = np.array(np.random.random(len(self.stock_data.keys())))
            weights = weights / np.sum(weights)
            all_weights[ind, :] = weights

            ret_arr[ind], vol_arr[ind], sharpe_arr[ind] = self.calculate_portfolio_performance(weights)

        plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Portfolio Optimization with Sharpe Ratio')
        plt.show()

    def minimize_volatility(self):
        num_stocks = len(self.stock_data)
        args = (self.expected_returns, self.covariance_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_stocks))
        result = minimize(self.negative_sharpe, num_stocks * [1. / num_stocks, ], args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x

    def negative_sharpe(self, weights):
        return -self.calculate_portfolio_performance(weights)[2]  # Return negative Sharpe for minimization