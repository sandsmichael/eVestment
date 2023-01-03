import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

fp = "C:\data\MS Equity Manager Compare eVestment.xlsx"
df = pd.read_excel(fp, sheet_name = 'General Info', skiprows = 4)
df['Full Name'] = df['Firm Name'].astype(str) + ' | ' + df['Product Name'].astype(str)
df['Custom Name'] = df['Firm: Firm Short Name'].astype(str) + ' | ' + df['Product Name'].astype(str).apply(lambda x : x[0:15]) 
df.set_index('Custom Name', inplace = True)

# Replace any missing datapoints and drop rows that are all NaN
df = df.replace('---',np.nan).dropna(how='all', axis=0)
# Casts all columns to strings and replaces text with NaN
for c in df.columns:
    df.loc[df[c].astype(str).str.contains('Not available in USD.', na=False), c] = np.nan

for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c])
    except:
        pass

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Columns                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
# Trailing Returns
returns_columns = [ c for c in df.columns if 'Returns - ' in c and 'in' not in c ][:-36] # Last 36 return columns are individual monthly series'
returns_column_one_month = [c for c in returns_columns if 'Month' in c][0]
returns_column_mrq = [c for c in returns_columns if 'MRQ' in c][0]
returns_column_ytd = [c for c in returns_columns if 'YTD' in c][0]
returns_column_one_year = [c for c in returns_columns if '1 Year' in c][0]
returns_column_three_year = [c for c in returns_columns if '3 Year' in c][0]
returns_column_five_year = [c for c in returns_columns if '5 Year' in c][0]
returns_column_ten_year = [c for c in returns_columns if '10 Year' in c][0]

first_quartile_returns_columns = [c for c in df.columns if 'Number in 1st Quartile' in c]
first_quartile_returns_column_mrq = [c for c in first_quartile_returns_columns if 'MRQ' in c][0]
first_quartile_returns_column_one_year = [c for c in first_quartile_returns_columns if '1 Year' in c][0]
first_quartile_returns_column_three_year = [c for c in first_quartile_returns_columns if '3 Year' in c][0]
first_quartile_returns_column_five_year = [c for c in first_quartile_returns_columns if '5 Year' in c][0]

excess_returns_columns = [c for c in df.columns if 'Excess Returns - ' in c and 'Down' not in c]
excess_returns_column_mrq = [c for c in excess_returns_columns if 'MRQ' in c][0]
excess_returns_column_one_year = [c for c in excess_returns_columns if '1 Year' in c][0]
excess_returns_column_three_year = [c for c in excess_returns_columns if '3 Year' in c][0]
excess_returns_column_five_year = [c for c in excess_returns_columns if '5 Year' in c][0]


monthly_return_columns = [ c for c in df.columns if 'Returns - ' in c and 'in' not in c ][-36:]



sharpe_columns = [ c for c in df.columns if 'Sharpe Ratio' in c]
sharpe_column_mrq = [c for c in sharpe_columns if 'MRQ' in c][0]
sharpe_column_one_year = [c for c in sharpe_columns if '1 Year' in c][0]
sharpe_column_three_year = [c for c in sharpe_columns if '3 Year' in c][0]
sharpe_column_five_year = [c for c in sharpe_columns if '5 Year' in c][0]


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Tear Sheet                                                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

monthly_returns = df[returns_columns].T.iloc[::-1]
monthly_returns.index = [c.rsplit(' - ')[-1].replace(')', '') for c in monthly_returns.index]


product_name = 'AGF Investments | AGF U.S. Large-'
product_returns = monthly_returns[product_name].to_frame().reset_index()
# product_returns.rename(columns={product_name:'returns', 'index':'Date'}, inplace = True)
# product_returns['returns'] = product_returns['returns'].round(2)

# product_returns['Date'] = pd.to_datetime(product_returns['Date'], format = "%m/%Y")
# product_returns.set_index('Date', inplace = True)
# product_returns['Year'] = product_returns.index.strftime('%Y')
# product_returns['Month'] = product_returns.index.strftime('%b')
# product_returns = product_returns.pivot('Year', 'Month', 'returns').fillna(0)
# grouped_returns = product_returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
# sns.heatmap(grouped_returns, annot=True, center=0,
#                 fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn' )
# plt.show()



data = df[[excess_returns_column_mrq, excess_returns_column_one_year, excess_returns_column_three_year, excess_returns_column_five_year]]
data = data[data.index == product_name].sort_values(by = [excess_returns_column_three_year], ascending = False)
data.columns = [c.split(' using ')[0] for c in data.columns]
print(data)

 
data = df[[sharpe_column_mrq, sharpe_column_one_year, sharpe_column_three_year, sharpe_column_five_year]]
data = data[data.index == product_name].sort_values(by = [ sharpe_column_three_year], ascending = False)
data.columns = [c.split(' using ')[0] for c in data.columns]
print(data)