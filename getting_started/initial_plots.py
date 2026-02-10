from data.fetch_data import get_data
import matplotlib as plt
import os

def make_plot(var1, var2):
    df, target_name = get_data()

    os.makedirs("plots", exist_ok = true)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=var1,
        y=var2,
        hue=target_name,
        style=target_name,
        s=90
    )


    plt.title(f"Fertility Diagnosis: {var1} vs {var2}")
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.grid(True)
    plt.savefig(f'data/plots/{var1}_v_{var2}.png', dpi=150)
    plt.close()


make_plot('age','accident')