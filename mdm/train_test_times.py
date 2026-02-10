import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Algorithm': ['PCC*', 'iForest*', 'KNN*', 'GlobalSTD3*', 'DC-VAE-ESA STD5*', 
                  'Telemanom-ESA*', '$\mathbf{LFTSAD}$', '$\mathbf{Matrix}$ $\mathbf{Profile}$', '$\mathbf{Anomaly}$ $\mathbf{Transformer}$'],
    'Mission 1 Training': [90, 655, 3844, 101, 13466, 13115, 568, np.nan, 6462],
    'Mission 1 Testing': [124, 393, 1233, 178, 5251, 6931, 2397, 5173, 5320],
    'Mission 2 Training': [63, 345, 4754, 60, 12440, 19725, 229, np.nan, 6462],
    'Mission 2 Testing': [73, 199, 21673, 95, 3068, 4666, 1041, 6896, 5320]
}

df = pd.DataFrame(data)


df_melted = df.melt(id_vars='Algorithm', 
                    var_name='Metric', 
                    value_name='Time (s)')



# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

colors = ["#0144FB", "#638BFB", "#FB9302", "#F7C680"]

g = sns.barplot(data=df_melted, 
            x='Algorithm', 
            y='Time (s)', 
            hue='Metric',
            palette=colors,
            ax=ax)

g.set_yscale('log')
# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Adjust legend
plt.legend(title='Metric')

# Tight layout to prevent label cutoff


plt.title('Training and Testing Times for Algorithms on Lightweight Subsets of Missions 1 and 2')
plt.tight_layout()
plt.savefig('train_test_times.pdf')