# https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/07_linear_models/03_preparing_the_model_data.ipynb
# https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/07_linear_models/05_predicting_stock_returns_with_linear_regression.ipynb

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr, spearmanr
from talib import RSI, BBANDS, MACD, ATR
import seaborn as sns
import matplotlib.pyplot as plt

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Parse                                                                                                            │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
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

# print(df.head())
# print(df.columns)


returns_columns = [ c for c in df.columns if 'Returns - ' in c and 'in' not in c ][-36:] # Last 36 return columns are individual monthly series'
monthly_returns = df[returns_columns].T.iloc[::-1]
monthly_returns.index = [c.rsplit(' - ')[-1].replace(')', '') for c in monthly_returns.index]
print(monthly_returns)
# print(monthly_returns.columns.tolist())


product_name = 'AGF Investments | AGF U.S. Large-'
data = monthly_returns[product_name].to_frame().reset_index()
data.rename(columns={product_name:'returns'}, inplace = True)
data['price'] = data['returns'].divide(100).add(1).cumprod() # Create a series of prices starting at $1 based on return series



data['SMA3'] = data['returns'].rolling(3).mean()
data['SMA6'] = data['returns'].rolling(6).mean()
data['SMA12'] = data['returns'].rolling(12).mean()


data['rsi'] = RSI(data['price'].to_numpy(), 6)

# ax = sns.distplot(data.rsi.dropna())
# ax.axvline(30, ls='--', lw=1, c='k')
# ax.axvline(70, ls='--', lw=1, c='k')
# ax.set_title('RSI Distribution with Signal Threshold')
# plt.tight_layout();
# plt.show()

def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=6)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)
# fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
# sns.distplot(data.loc[data.price<100, 'bb_low'].dropna(), ax=axes[0])
# sns.distplot(data.loc[data.price<100, 'bb_high'].dropna(), ax=axes[1])
# plt.tight_layout();
# plt.show()
data = data.merge(compute_bb(data['price']), left_index=True, right_index = True, how = 'inner')
# data['bb_high'] = data.bb_high.sub(data.price).div(data.bb_high).apply(np.log1p)
# data['bb_low'] = data.price.sub(data.bb_low).div(data.price).apply(np.log1p)

# def compute_macd(close):
#     macd = MACD(close)[0]
#     return ((macd - np.mean(macd))/np.std(macd)).to_frame().rename(columns = {0: 'macd'})
# data = data.merge(compute_macd(data['price']), left_index=True, right_index = True, how = 'inner')
# print(data.macd.describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999]).apply(lambda x: f'{x:,.1f}'))

lags = [1, 3, 6, 12]
percentiles=[.0001, .001, .01]
percentiles+= [1-p for p in percentiles]
# data.returns.describe(percentiles=percentiles).iloc[2:].to_frame('percentiles').style.format(lambda x: f'{x:,.2%}')

# Winsorize outliers
q = 0.0001
for lag in lags:
    data[f'return_{lag}m'] = (data.price.pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(q),
                                                       upper=x.quantile(1 - q)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1)
                                )
# shift lagged returns
# for t in [1, 2, 3, 4, 5]:
#     for lag in [1, 3, 6]:
#         data[f'return_{lag}d_lag{t}'] = (data[f'return_{lag}d'].shift(t * lag))

# # Forward returns
for t in [1, 3, 6, 12]:
    data[f'target_{t}m'] = data[f'return_{t}m'].shift(-t)

# data['year'] = data.index.get_level_values('date').year
# data['month'] = data.index.get_level_values('date').month



# explore data:
y = data.filter(like='target').fillna(0)
X = data.drop(y.columns, axis=1)

# sns.clustermap(y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt='.2%');
# plt.show()

print(data)
print(X)
print(X.shape)
from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

target = 'target_1m'
X = (X.drop(['index', 'returns', 'price'], axis=1)
     .transform(lambda x: (x - x.mean()) / x.std())
    .fillna(0))
print(X)
print(X.shape)

model = OLS(endog=y[target], exog=add_constant(X))
trained_model = model.fit()
print(trained_model.summary()) 

Xnew = pd.DataFrame([1] + X.iloc[-1].values.tolist()).transpose() 
ypred = trained_model.predict(Xnew)
print(ypred)
# preds = trained_model.predict(add_constant(X))
# residuals = y[target] - preds
# fig, axes = plt.subplots(ncols=2, figsize=(14,4))
# sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
# axes[0].set_title('Residual Distribution')
# axes[0].legend()
# plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
# axes[1].set_xlabel('Lags')
# sns.despine()
# fig.tight_layout();
# plt.show()





# print(data)
