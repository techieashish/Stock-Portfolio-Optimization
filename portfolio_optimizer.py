import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.returns = self.calculate_daily_returns()

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
        expected_returns, covariance_matrix = self.get_expected_returns_and_covariance()
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def minimize_volatility(self):
        num_stocks = len(self.stock_data)
        initial_weights = np.array([1 / num_stocks] * num_stocks)
        bounds = tuple((0, 1) for _ in range(num_stocks))
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        result = minimize(lambda weights: self.calculate_portfolio_performance(weights)[1],
                          initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def optimize_portfolio(self):
        optimal_weights = self.minimize_volatility()
        return optimal_weights
