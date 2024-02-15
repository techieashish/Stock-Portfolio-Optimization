import pandas as pd
import numpy as np
from scipy.optimize import minimize


class MarkPortfolioOptimizer:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.returns = self.calculate_daily_returns()
        self.expected_returns, self.covariance_matrix = self.get_expected_returns_and_covariance()

    def calculate_daily_returns(self):
        returns = {}
        for stock, data in self.stock_data.items():
            prices = pd.Series([float(v[4]) if v[4] != '#VALUE!' else np.nan for v in data.values()],
                               index=pd.to_datetime(list(data.keys()), format='%d-%m-%Y'))
            daily_returns = prices.pct_change().fillna(0)
            returns[stock] = daily_returns
        daily_returns_df = pd.DataFrame(returns)
        print("\nDaily Returns:")
        print(daily_returns_df.round(4).to_string())
        return daily_returns_df

    def get_expected_returns_and_covariance(self):
        expected_returns = self.returns.mean()
        covariance_matrix = self.returns.cov()
        return expected_returns, covariance_matrix

    def calculate_portfolio_performance(self, weights):
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def minimize_volatility(self):
        num_stocks = len(self.stock_data)
        initial_weights = np.array([1. / num_stocks] * num_stocks)
        bounds = tuple((0, 1) for _ in range(num_stocks))
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

        result = minimize(lambda weights: self.calculate_portfolio_performance(weights)[1],
                          initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x

    def print_optimized_weights(self, optimal_weights):
        print("\nOptimized Portfolio Weights:")
        for stock, weight in zip(self.stock_data.keys(), optimal_weights):
            print(f"{stock}: {weight * 100:.2f}%")

    def write_to_excel(self, optimal_weights, file_name='portfolio_optimization_results.xlsx'):
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:

            self.returns.round(4).to_excel(writer, sheet_name='Daily Returns')


            weights_df = pd.DataFrame(optimal_weights, index=self.stock_data.keys(), columns=['Optimized Weights'])
            weights_df['Optimized Weights'] = weights_df['Optimized Weights'].apply(lambda x: f"{x * 100:.2f}%")
            weights_df.to_excel(writer, sheet_name='Optimized Weights')

            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for col in worksheet.columns:
                    max_length = max((len(str(cell.value)) for cell in col))
                    worksheet.column_dimensions[col[0].column_letter].width = max_length + 2

        print(f"Portfolio optimization results written to '{file_name}'")

