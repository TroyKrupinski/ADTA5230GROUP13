#Author: Troy Krupinski

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the matplotlib figure
plt.figure(figsize=(15, 12))

# Variables of interest for visualization
variables_to_plot = ['ownd', 'kids', 'inc', 'sex', 'wlth', 'hv', 'incmed', 'incavg', 'low', 'npro', 'gifdol', 'gifl', 'gifr', 'mdon', 'lag', 'gifa']

# Create subplots for each variable
for i, var in enumerate(variables_to_plot):
    plt.subplot(4, 4, i + 1)
    sns.histplot(data[var], kde=False, bins=30)
    plt.title(var)
    plt.tight_layout()

plt.show()