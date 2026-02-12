import os
from data.fetch_data import get_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def make_plot(var1, var2, var3):
    df, target_name = get_data()

    target_colors = list(map(lambda s: 'w' if s=='N' else 'r', df[target_name]))
    # print(target_colors)
    os.makedirs("getting_started/plots", exist_ok = True)

    sns.set_style("whitegrid", {'axes.grid' : False})

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111, projection='3d')

    x = var1
    y = var2
    z = var3
    ax.scatter(x, y, z, c=target_colors, data=df, marker='o')
    # cmap='RdYlGn_r'
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_zlabel(var3)

    plt.title(f"Fertility Diagnosis: {var1} vs {var2} vs {var3}")
    
    plt.savefig(f'getting_started/plots/{var1}_v_{var2}_v_{var3}.png', dpi=150)
    plt.close()


make_plot('accident', 'surgical_intervention', 'smoking')
make_plot('high_fevers', 'smoking', 'child_diseases')
make_plot('season', 'alcohol', 'surgical_intervention')