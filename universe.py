""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Analysis of an eVestment Universe aimed at flagging interesting products for more detailed, product level        │
  │ analysis                                                                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects

sns.set_style("whitegrid")


def add_median_labels(ax, fmt='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


fig = plt.figure(figsize=(15, 10), constrained_layout=True) # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html
gs = GridSpec(4, 4, figure=fig)

ax3 = fig.add_subplot(gs[:2, -2:])   
ax2 = fig.add_subplot(gs[:2, :-2])

ax4 = fig.add_subplot(gs[2:, :-2])
ax1 = fig.add_subplot(gs[2:, -2:])


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

print(df.head())
print(df.columns)


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
monthly_return_columns = [ c for c in df.columns if 'Returns - ' in c and 'in' not in c ][-36:]


sharpe_columns = [ c for c in df.columns if 'Sharpe Ratio' in c]
sharpe_column_mrq = [c for c in sharpe_columns if 'MRQ' in c][0]
sharpe_column_one_year = [c for c in sharpe_columns if '1 Year' in c][0]
sharpe_column_three_year = [c for c in sharpe_columns if '3 Year' in c][0]

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
    # one_month_leaders = get_leaders(df = df, col = returns_column_one_month, n = 5)
    mrq_leaders = get_leaders(df = df, col = returns_column_mrq, n = 10)
    one_year_leaders = get_leaders(df = df, col = returns_column_one_year, n = 10)
    three_year_leaders = get_leaders(df = df, col = returns_column_three_year, n = 10)
    five_year_leaders = get_leaders(df = df, col = returns_column_five_year, n = 10)

    returns = get_returns(df)
    trailing_leaders_names = get_unique(dfs = [mrq_leaders, one_year_leaders, three_year_leaders, five_year_leaders])
    trailing_leaders = returns[returns.index.isin(trailing_leaders_names)].sort_values(by = returns_column_mrq, ascending = False)[[returns_column_mrq, returns_column_one_year, returns_column_three_year, returns_column_five_year]]
    
    return trailing_leaders


def combined_sharpe_leaders():
    mrq_leaders = get_leaders(df = df, col = sharpe_column_mrq, n = 10)
    one_year_leaders = get_leaders(df = df, col = sharpe_column_one_year, n = 10)
    three_year_leaders = get_leaders(df = df, col = sharpe_column_three_year, n = 10)

    sharpe_leaders_names = get_unique(dfs = [mrq_leaders, one_year_leaders, three_year_leaders])
    sharpe_leaders = df[df.index.isin(sharpe_leaders_names)].sort_values(by = returns_column_mrq, ascending = False)[[sharpe_column_mrq, sharpe_column_one_year, sharpe_column_three_year]]
    
    return sharpe_leaders

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Analysis                                                                                                │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

#Bar chart
# fig, ax = plt.subplots(figsize=(15,10))
combined_trailing_leaders = combined_trailing_leaders().dropna(subset = [returns_column_mrq])
combined_trailing_leaders.columns = [c.split('(')[0] for c in combined_trailing_leaders.columns]

# print(combined_trailing_leaders)
# melt = combined_trailing_leaders.reset_index().melt(id_vars = 'Custom Name', var_name = 'Trailing Period', value_name = 'Returns')
# sns.barplot(data = melt, x = 'Custom Name', y = 'Returns', hue = 'Trailing Period', dodge = True,  ci=None, palette="Set2",  ax = ax1,)
# ax1.set_xticklabels( ax1.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize = 6)
# ax1.set_title('Top 5 Products from each Trailing Period')
# ax1.set_xlabel('xlabel', fontsize=6)
# # plt.subplots_adjust(left=None, bottom=0.5, right=None, top=None, wspace=None, hspace=None)

#Heat map
# fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(combined_trailing_leaders, ax=ax2, annot=True, center=0,
                fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn', annot_kws={"fontsize":7}, xticklabels=True, yticklabels=True, cbar=False )
ax2.set_title('Top 10 Products from each Trailing Period')
ax2.set_xlabel('xlabel', fontsize=8)
ax2.set_ylabel('xlabel', fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=7)
ax2.set_xticklabels( ax2.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize = 6)


# Sharpe
# combined_sharpe_leaders = combined_sharpe_leaders().dropna(subset = [sharpe_column_mrq])
# combined_sharpe_leaders.columns = [c.split('(')[0] for c in combined_sharpe_leaders.columns]
# #Heat map
# # fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(combined_sharpe_leaders, ax=ax1, annot=True, center=0,
#                 fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn', annot_kws={"fontsize":7}, xticklabels=True, yticklabels=True, cbar=False )
# ax1.set_title('Top 10 Products from each Trailing Period')
# ax1.set_xlabel('xlabel', fontsize=8)
# ax1.set_ylabel('xlabel', fontsize=8)
# ax1.tick_params(axis='both', which='major', labelsize=7)
# ax1.set_xticklabels( ax2.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize = 6)


# box plot
box = df[[returns_column_mrq, returns_column_one_year, returns_column_three_year, returns_column_five_year]]
box.columns = [c.split('(')[0] for c in box.columns]
melt = box.reset_index().melt(id_vars = 'Custom Name', var_name = 'Trailing Period', value_name = 'Returns')
sns.boxplot(data = melt, x = 'Trailing Period', y = 'Returns', ax = ax1)
ax1.set_title('Trailing Period Returns')
ax1.set_xlabel('xlabel', fontsize=8)
ax1.set_ylabel('xlabel', fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=7)
add_median_labels(ax1)



# Sum of ranks
ranks_sum = df.reset_index()[['Full Name',  returns_column_mrq, returns_column_one_year, returns_column_three_year, returns_column_five_year]].set_index('Full Name')
frames = []
for c in ranks_sum.columns:
    tmp = ranks_sum[c].to_frame().sort_values(by = c, ascending = False).dropna(axis=0)
    tmp[f'{c} Rank'] = [i+1 for i in range(len(tmp))]
    frames.append(tmp[f'{c} Rank'].to_frame())

from functools import reduce
ranks = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how = 'outer'), frames)
custom_names = df.reset_index()[['Full Name', 'Custom Name']].set_index('Full Name')
ranks = ranks.merge(custom_names, left_index = True, right_index = True, how = 'inner')
ranks.reset_index(drop=True, inplace = True)
ranks.set_index('Custom Name', inplace = True)
ranks.columns = [c.split('(')[0] for c in ranks.columns]
ranks['sum'] = ranks.sum(axis=1)
ranks = ranks.sort_values(by = 'sum', ascending = True)
ranks = ranks.iloc[:15]
# fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(ranks, ax=ax3, annot=True,
                fmt="0.0f", linewidths=0.5, cmap = ListedColormap(['white']), cbar=False )
ax3.set_title('Trailing Ranks Sum')
ax3.set_xlabel('xlabel', fontsize=8)
ax3.set_ylabel('xlabel', fontsize=8)
ax3.tick_params(axis='both', which='major', labelsize=7)
ax3.set_xticklabels( ax3.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize = 6)
# render_mpl_table(ranks.round(0).reset_index(), header_columns=0,  font_size=7, ax=ax3)





# #Heat map
# returns = get_returns(df)
# mrq_leaders = returns[returns.index.isin(get_leaders(df = df, col = returns_column_mrq, n =10).index)].sort_values(by = returns_column_mrq, ascending = False)
# print(mrq_leaders)
# # fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(mrq_leaders, ax=ax4, annot=True, center=0,
#                 fmt="0.1f", linewidths=0.5, cmap = 'RdYlGn' )
# ax4.set_title('Top Products 1 Quarter Trailing Returns')
# ax4.set_xlabel('xlabel', fontsize=8)
# ax4.set_ylabel('xlabel', fontsize=8)
# plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)


number_of_first_quartile_returns = df[[first_quartile_returns_column_mrq, first_quartile_returns_column_one_year, first_quartile_returns_column_three_year, first_quartile_returns_column_five_year]]#.sort_values(by = first_quartile_returns_column_one_year, ascending = False).iloc[:10]
number_of_first_quartile_returns = number_of_first_quartile_returns[number_of_first_quartile_returns[first_quartile_returns_column_one_year] >= 8]
number_of_first_quartile_returns.columns = [c.split('(')[0].split(' - ')[1] for c in number_of_first_quartile_returns.columns]

medians = number_of_first_quartile_returns.median()
melt = number_of_first_quartile_returns.reset_index().melt(id_vars = 'Custom Name', value_name='Number in 1st Qrtl', var_name='Trailing Period')
sns.scatterplot(data = melt, x = 'Number in 1st Qrtl', y = 'Custom Name', hue = 'Trailing Period', ax=ax4, )
for i in medians.values:
    ax4.axvline(i, color = 'black')
ax4.set_title('Number of Returns in 1st Quartile. Vertical line is Median. >=8 Trailing 1 Year')
ax4.set_xlabel(None)
# ax4.set_ylabel('xlabel', fontsize=8)
ax4.tick_params(axis='both', which='major', labelsize=7)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

# # 3 Year excess return
# excess_returns = df[[excess_returns_column_mrq, excess_returns_column_one_year, excess_returns_column_three_year]]
# excess_returns_mrq_leaders = get_leaders(df, col = excess_returns_column_mrq, n = 5)
# excess_returns_one_year_leaders = get_leaders(df, col = excess_returns_column_one_year, n = 5)
# excess_returns_three_year_leaders = get_leaders(df, col = excess_returns_column_three_year, n = 5)
# excess_returns_leaders_names = get_unique(dfs = [excess_returns_mrq_leaders, excess_returns_one_year_leaders, excess_returns_three_year_leaders])
# excess_returns_leaders = excess_returns[excess_returns.index.isin(excess_returns_leaders_names)].sort_values(by = excess_returns_column_one_year, ascending = False).iloc[:10]
# print(excess_returns_leaders)



#Var




#Monthly regression - Dependent vars = Benchmark Return, Interest rate change
monthly_returns = df[monthly_return_columns].T.iloc[::-1]

# import yfinance as yf
# index =  yf.download('SPY', start=self.START_DATE, end=self.END_DATE, progress = False)




plt.show()



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Drawdowns                                                                                                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
# df = pd.read_excel(fp, sheet_name = 'Largest Drawdowns')
# blank_rows = np.where(df.isna().all(axis=1))[0]
# print(blank_rows)
