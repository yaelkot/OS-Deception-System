import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os


def sns_plots(df, x, y):

    folder = "eight_features_plot"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    out_file_name = folder + "/" + y + ".png"

    # font = {'family': 'normal',
    font = {'size': 12}

    plt.rc('font', **font)
    plt.rcParams["axes.labelsize"] = 20
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.2)

    fig = plt.figure(figsize=(34, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    # ax4 = fig.add_subplot(gs[0, 3])

    # sns.boxplot(x=x, y=y, data=df, ax=ax1)
    sns.violinplot(x=x, y=y, data=df, ax=ax1)
    sns.scatterplot(x=x, y=y, data=df, ax=ax2)
    sns.histplot(x=x, y=y, data=df, ax=ax3)
    fig.savefig(out_file_name)


if __name__ == "__main__":
    df = pd.read_csv(".\\real-traffic\\train-set.csv")

    # filtering out one distant outlier
    newdf = df[df["tcp.window_size"] < 2000000]

    feature_list = ['ip.ttl', 'ip.len', 'tcp.srcport', 'tcp.dstport', 'tcp.ack', 'tcp.hdr_len',
                    'tcp.flags.ack', 'tcp.window_size', 'tcp.time_delta']
    for feature in feature_list:
        sns_plots(newdf, "os", feature)
