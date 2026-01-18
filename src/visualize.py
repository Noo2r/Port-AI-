import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_trajectory(df, mmsi, savefig=None):
    ship = df[df["MMSI"]==mmsi].sort_values("TSTAMP")
    plt.figure(figsize=(8,6))
    plt.plot(ship["LONGITUDE"], ship["LATITUDE"], marker="o")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"Trajectory MMSI {mmsi}")
    if savefig:
        plt.savefig(savefig)
    plt.show()

def plot_heatmap(df, savefig=None):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x="LONGITUDE", y="LATITUDE", data=df.sample(min(20000,len(df))), s=1)
    plt.title("AIS Traffic Sample")
    if savefig:
        plt.savefig(savefig)
    plt.show()
