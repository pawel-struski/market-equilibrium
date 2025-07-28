import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# select the experiment for which you want to plot
EXPERIMENT_ID = 5

# set the paths
results_path = Path(__file__).parent.resolve() / "results"
figures_path = Path(__file__).parent.resolve() / "plots"

# read the results data
df_data = pd.read_csv(results_path/f'experiment_{EXPERIMENT_ID}.csv', index_col=0)

# plot the offer prices
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_data.index, y='price', marker='x', data=df_data)
plt.axhline(y=2, color='red', linestyle='--')
for x in [5, 10, 15, 20]:
    plt.axvline(x=x-0.5, color='grey', linestyle='--', linewidth=1.5)
plt.xticks(ticks=df_data.index, labels=df_data['iteration'], rotation=0)
plt.title('Offer price at each iteration')
plt.xlabel('Iteration')
plt.ylabel('Price')
plt.tight_layout()
plt.ylim(0, 4)
plt.savefig(figures_path/f'offer_prices_{EXPERIMENT_ID}.png', dpi=300, bbox_inches='tight')

# plot the transaction prices
df_plot = df_data.loc[df_data['transaction']==True].reset_index()
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_plot.index, y='price', marker='x', 
                data=df_plot)
plt.axhline(y=2, color='red', linestyle='--')
last_indices_in_round = df_plot.groupby('round')['iteration'].apply(lambda g: g.index.max())
for x in last_indices_in_round[:-1]:
    plt.axvline(x=x+0.5, color='grey', linestyle='--', linewidth=1.5)
plt.xticks(ticks=df_plot.index, labels=df_plot['iteration'], rotation=0)
plt.title('Transaction prices')
plt.xlabel('Iteration')
plt.ylabel('Price')
plt.tight_layout()
plt.ylim(0, 4)
plt.savefig(figures_path/f'transaction_prices_{EXPERIMENT_ID}.png', dpi=300, bbox_inches='tight')


