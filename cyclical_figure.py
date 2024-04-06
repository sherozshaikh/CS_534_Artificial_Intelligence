import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_encoded_sin_cos = pd.read_csv('cleaned_transactions (1).csv')

    fig, ax = plt.subplots(figsize=(7, 5))
    sp = ax.scatter(
        data_encoded_sin_cos["Hour_sin"],
        data_encoded_sin_cos["Hour_cos"],
        c=data_encoded_sin_cos["Hour"],
        cmap='plasma'
    )
    ax.set(
        xlabel="sin(Hour)",
        ylabel="cos(Hour)",
    )
    _ = fig.colorbar(sp)
    data_encoded_sin_cos = data_encoded_sin_cos.drop(columns='Hour')

    plt.show()

if __name__ == '__main__':
    main()