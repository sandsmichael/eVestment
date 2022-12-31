import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Parse                                                                                                            │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
fp = "C:\data\MS Equity Manager Compare eVestment.xlsx"
df = pd.read_excel(fp, sheet_name = 'General Info', skiprows = 4)
df['Custom Name'] = df['Firm: Firm Short Name'].astype(str) + df['Product Name'].astype(str).apply(lambda x : x[0:15]) 
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

print(df.head())
print(df.columns)


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Columns                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
returns_columns = [ c for c in df.columns if 'Returns - ' in c and 'in' not in c ]
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



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Helpers                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
def get_returns(df):
    returns_df = df[returns_columns]
    return returns_df


def get_leaders(df, col, n=10):
    return df[col].to_frame().sort_values( by = col, ascending=False).dropna().iloc[:n]


def get_unique(dfs:list):
    index_values = []
    for df in dfs:
        index_values = index_values + [*df.index.values]
    return index_values


def combined_trailing_leaders():
    """ Gets top n leaders from the 1 month, MRQ, 1 Year and 3 Year trailing periods. Keeps unique products only. Returns a combined dataframe of all trailing periods.
    """
    one_month_leaders = get_leaders(df = df, col = returns_column_one_month, n = 5)
    mrq_leaders = get_leaders(df = df, col = returns_column_mrq, n = 5)
    one_year_leaders = get_leaders(df = df, col = returns_column_one_year, n = 5)
    three_year_leaders = get_leaders(df = df, col = returns_column_three_year, n = 5)

    returns = get_returns(df)
    trailing_leaders_names = get_unique(dfs = [one_month_leaders, mrq_leaders, one_year_leaders, three_year_leaders])
    trailing_leaders = returns[returns.index.isin(trailing_leaders_names)].sort_values(by = returns_column_mrq, ascending = False)[[returns_column_one_month, returns_column_mrq, returns_column_one_year, returns_column_three_year]]
    
    return trailing_leaders


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Analysis                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

#Bar chart
# fig, ax = plt.subplots(figsize=(15,10))
# combined_trailing_leaders = combined_trailing_leaders()
# melt = combined_trailing_leaders.reset_index().melt(id_vars = 'Custom Name', var_name = 'Trailing Period', value_name = 'Returns')
# sns.barplot(data = melt, x = 'Custom Name', y = 'Returns', hue = 'Trailing Period', dodge = True,  ci=None, palette="Set2",  ax = ax,)
# ax.set_xticklabels( ax.get_xticklabels(), rotation=90, horizontalalignment='center')
# ax.set_title('Top 5 Products from any Trailing Period')
# plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None)
# plt.show()

# #Heat map
# fig, ax = plt.subplots(figsize=(15,10))
# ax = sns.heatmap(combined_trailing_leaders, ax=ax, annot=True, center=0,
#                 fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn' )
# ax.set_title('Top 5 Products from any Trailing Period')
# plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
# plt.show()

# #Heat map
# returns = get_returns(df)
# one_month_leaders = returns[returns.index.isin(get_leaders(df = df, col = returns_column_one_month, n =10).index)].sort_values(by = returns_column_one_month, ascending = False)
# fig, ax = plt.subplots(figsize=(15,10))
# ax = sns.heatmap(one_month_leaders, ax=ax, annot=True, center=0,
#                 fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn' )
# ax.set_title('Top Products 1 Month Trailing Returns')
# plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
# plt.show()


number_of_first_quartile_returns = df[[first_quartile_returns_column_mrq, first_quartile_returns_column_one_year, first_quartile_returns_column_three_year, first_quartile_returns_column_five_year]]#.sort_values(by = first_quartile_returns_column_one_year, ascending = False).iloc[:10]
number_of_first_quartile_returns = number_of_first_quartile_returns[number_of_first_quartile_returns[first_quartile_returns_column_one_year] >= 8]
medians = number_of_first_quartile_returns.median()
print(medians)
melt = number_of_first_quartile_returns.reset_index().melt(id_vars = 'Custom Name', value_name='Number in 1st Qrtl', var_name='Trailing Period')
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.scatterplot(data = melt, x = 'Number in 1st Qrtl', y = 'Custom Name', hue = 'Trailing Period', ax=ax, )
for i in medians.values:
    ax.axvline(i, color = 'black')
ax.set_title('Products with >= 8 Returns in 1st Quartile over Trailing 1-Year. Horizontal line is Median.')
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.show()




# fig = plt.figure(figsize=(15, 10), constrained_layout=True) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html
# gs = GridSpec(4, 4, figure=fig)
# ax1 = fig.add_subplot(gs[0, : -3])        
# ax2 = fig.add_subplot(gs[1, :-3])
# ax3 = fig.add_subplot(gs[2, :-3])
# ax4 = fig.add_subplot(gs[3, :-3])

# ax5 = fig.add_subplot(gs[0, 2:])
# ax6 = fig.add_subplot(gs[1, 2:])
# ax7 = fig.add_subplot(gs[2, 2:])
# ax8 = fig.add_subplot(gs[3, 1:2])
# ax9 = fig.add_subplot(gs[3, 2:3])
# ax10 = fig.add_subplot(gs[3, 3:])

# ax11 = fig.add_subplot(gs[0, 1:2])




""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Drawdowns                                                                                                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
# df = pd.read_excel(fp, sheet_name = 'Largest Drawdowns')
# blank_rows = np.where(df.isna().all(axis=1))[0]
# print(blank_rows)
