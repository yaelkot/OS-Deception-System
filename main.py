### Example Inputs ###
# python main.py -file thursday-100M-v2.csv -features 4,5,6,8,20,22,23,24,25,26,27,29 -label 32 -test 0.2
# python main.py -file thursday-100M-v2.csv -features 1,5,8,13,17,18,19,20,22,23,24,25,26,29 -label 32 -test 0.2
### End ###


import pandas as pd
import sys

import sklearn

from ML_Models_Structure import ClassifierModel
from traffic_data_preprocess import data_preprocess, add_label


# Consider all Linux versions as Linux
# ip_dict = {
#     '10.0.0.5': 'Linux',
#     '10.0.0.6': 'Win 7',
#     '192.168.0.100': 'Win 10',
#     '192.168.1.105': 'Win 11',
#     '192.168.1.34': 'Win 11',
#     '192.168.1.81': 'Mac 2017',
#     '192.168.31.59': 'Mac 2018',
#     '192.168.1.56': 'Mac 2019',
#     '10.0.0.7': 'Linux',
#     '10.100.102.8': 'Linux'
# }

# Consider all Linux versions as Linux and Windows as Windows
ip_dict = {
     '10.0.0.5': 'Linux',     #was Ubuntu
     '10.0.0.6': 'Windows',
     '192.168.0.100': 'Windows',
     '192.168.1.105': 'Windows',
     '192.168.1.34': 'Windows',
     '192.168.1.81': 'MacOS',
     '192.168.31.59': 'MacOS',
     '192.168.1.56': 'MacOS',
     '10.0.0.7': 'Linux',     #was ARCH
     '10.100.102.8': 'Linux'  #was RHEL
}

# Consider all Windows as Windows
# ip_dict = {
#     '10.0.0.5': 'Ubuntu 20.4',
#     '10.0.0.6': 'Windows',
#     '192.168.0.100': 'Windows',
#     '192.168.1.105': 'Windows',
#     '192.168.1.34': 'Windows',
#     '192.168.1.81': 'Mac 2017'
#     '192.168.31.59': 'Mac 2018',
#     '192.168.1.56': 'Mac 2019',
#     '10.0.0.7': 'ARCH',
#     '10.100.102.8': 'RHEL8'
# }


####################################################################
########################## Option Set Up ###########################
####################################################################

def option_check():
    # all available argument options
    avail_options = ["-file", "-features", "-label", "-test"]

    # receive user given options
    options = [opt for opt in sys.argv[1:] if opt.startswith("-")]

    # receive user given arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    # raise error if user given option is wrong
    for i in options:
        if i not in avail_options:
            raise SystemExit(f"Usage: {sys.argv[0]} -file -features -label -test <arguments>...")

    # raise error if not all options or arguments are available
    if len(options) != 4 or len(args) != 4:
        raise SystemExit(f"Usage: {sys.argv[0]} -file -features -label -test <arguments>...")

    return args


####################################################################
########################## Main Function ###########################
####################################################################


if __name__ == "__main__":

    # # args = option_check()
    # # filename = args[0]
    #
    # filename = "dataset\\monday.csv"
    # df = pd.read_csv(filename)
    # processed_df = data_preprocess(df)
    # labeled_df = add_label(processed_df, ip_dict)
    #
    # print("Total number of packets: ", len(labeled_df))
    # print(labeled_df)
    #
    # labeled_df.to_csv("dataset\\labeled_monday.csv", encoding='utf-8', index=False)
    #
    #
    filename = ".\\labeled.csv"
    labeled_df = pd.read_csv(filename)
    print(labeled_df)
    # #
    # # # fingerprinting with classification
    # #
    # x_iloc_list = ['ip.ttl', 'ip.len', 'tcp.hdr_len', 'tcp.window_size', 'ip.flags.df', 'tcp.flags.syn',
    #                'ip.hdr_len', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.seq', 'tcp.len', 'tcp.time_delta']

    x_iloc_list = [8, 13, 20, 29, 5, 23, 1, 26, 25, 17, 19, 32]  # indexes in the labeled csv
    y_iloc = 33
    labeled_df = labeled_df.dropna()
    # nan_values = float('NaN')
    # non_numeric = 'not a numeric object'
    # labeled_df.replace('', non_numeric, inplace=True)
    # labeled_df.replace('', nan_values, inplace=True)
    # new_df = labeled_df['ip.ttl']
    testSize = float(0.2)
    model = ClassifierModel(labeled_df, x_iloc_list, y_iloc, testSize)

    filename = "dataset\\yael-pcap.csv"
    df = pd.read_csv(filename)
    df = df.drop(columns=['ip.tos', 'tcp.options.mss_val'])
    user_traffic = df.dropna()
    x_iloc_list = [8, 13, 20, 29, 5, 23, 1, 26, 25, 17, 19, 32]   # indexes in user's csv

    knn = model.KNN()
    model.SVM('linear')
    model.SVM('rbf')
    model.NB()
    model.RF()
    model.ANN()
    model.DT()

    # print(user_traffic)
    print()
    model.run_models(user_traffic, x_iloc_list, 'Windows')

    # sklearn.metrics.confusion_matrix(, user_traffic)