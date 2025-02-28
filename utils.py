import numpy as np
import pandas as pd
# Given AI chip market revenue data


df1 = pd.read_excel('data/statistic_id1283358_ai-chip-market-revenue-worldwide-2023-2025.xlsx', sheet_name='Data', header=0)

years = np.array(pd.to_numeric(df1.iloc[4:, 1].astype(str).str.extract('(\d+)')[0]))

revenue = np.array(pd.to_numeric(df1.iloc[4:, 2].astype(str).str.extract('(\d+)')[0]))
print(revenue)
def bass_model(t, p, q, M):
    """Bass Diffusion Model function."""
    return M * (1 - np.exp(-(p + q) * (t - years[0]))) / (1 + (q/p) * np.exp(-(p + q) * (t - years[0])))

def adopters_per_period(t, p, q, M):
    """Calculate the number of adopters at each time period."""
    ft = (p + q) ** 2 * np.exp(-(p + q) * (t - years[0])) * M / ((p + q * np.exp(-(p + q) * (t - years[0]))) ** 2)
    return ft