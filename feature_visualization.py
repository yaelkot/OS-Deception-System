import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import os


def sns_plots(df, x, y):
    folder = "sns_plots"
    if not os.path.isdir(folder):
        os.mkdir(folder)

    out_file_name = folder + "/" + y + ".png"

    # font = {'family': 'normal',
    font = {'size': 12}

    plt.rc('font', **font)
    plt.rcParams["axes.labelsize"] = 20
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)

    fig = plt.figure(figsize=(24, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[0, 2])

    sns.boxplot(x=x, y=y, data=df, ax=ax1)
    # sns.violinplot(x=x, y=y, data=df, ax=ax2)
    # sns.scatterplot(x=x, y=y, data=df, ax=ax3)
    sns.histplot(x=x, y=y, data=df, ax=ax2)
    # sns.displot(x=x, y=y, data=df)
    """ sns.displot(df["ip.len"], x="ip.len", bins=[1, 2, 3, 4, 5, 6, 7])
    sns.displot(df["ip.ttl"], x="ip.ttl", bins=[1, 2, 3, 4, 5, 6, 7])
    sns.displot(df["tcp.ack"], x="tcp.ack", bins=[1, 2, 3, 4, 5, 6, 7])
    sns.displot(df["tcp.hdr_len"], x="ip.len", bins=[1, 2, 3, 4, 5, 6, 7])
    sns.displot(df["tcp.window_size"], x="tcp.window_size", bins=[1, 2, 3, 4, 5, 6, 7])"""


    fig.savefig(out_file_name)


def dist_plots(df, x, y):

    # print(df["ip.len"][1])
    sns.kdeplot(x=x, y=y, data=df)


if __name__ == "__main__":
    df = pd.read_csv("dataset\\labeled_monday.csv")

    # filtering out one distant outlier
    newdf = df[df["tcp.window_size"] < 2000000]

    # feature_list = ["ip.ttl", "ip.len", "tcp.seq", "tcp.len", "tcp.hdr_len",
    #                 "tcp.window_size", "ip.flags.df", "tcp.seq", "tcp.ack", "tcp.flags.fin",
    #                 "tcp.flags.syn", "tcp.flags.reset", "tcp.flags.push"]
    feature_list = ["ip.ttl", "ip.len", "tcp.hdr_len",
                    "tcp.window_size", "tcp.ack"]
    for feature in feature_list:
        sns_plots(newdf, "os", feature)
        print(feature)
        # dist_plots(newdf, "os", feature)
