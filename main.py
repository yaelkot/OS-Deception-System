
### Example Inputs ###
# python main.py -file thursday-100M-v2.csv -features 4,5,6,8,20,22,23,24,25,26,27,29 -label 32 -test 0.2
# python main.py -file thursday-100M-v2.csv -features 1,5,8,13,17,18,19,20,22,23,24,25,26,29 -label 32 -test 0.2
### End ###


import pandas as pd
import sys
from ML_Models_Structure import ClassifierModel
from traffic_data_preprocess import data_preprocess, add_label

# known OS (source: https://www.unb.ca/cic/datasets/ids-2017.html)
ip_dict = {
    '192.168.10.51': 'Ubuntu server 12',
    '192.168.10.19': 'Ubuntu 14.4',
    '192.168.10.17': 'Ubuntu 14.4',
    '192.168.10.16': 'Ubuntu 16.4',
    '192.168.10.12': 'Ubuntu 16.4',
    '192.168.10.9': 'Win 7',
    '192.168.10.5': 'Win 8.1',
    '192.168.10.8': 'Win Vista',
    '192.168.10.14': 'Win 10',
    '192.168.10.15': 'Win 10',
    '192.168.10.25': 'macOS'
}

# Consider all Ubuntu versions as Ubuntu
# ip_dict = {
#     '192.168.10.51': 'Ubuntu',
#     '192.168.10.19': 'Ubuntu',
#     '192.168.10.17': 'Ubuntu',
#     '192.168.10.16': 'Ubuntu',
#     '192.168.10.12': 'Ubuntu',
#     '192.168.10.9': 'Win 7',
#     '192.168.10.5': 'Win 8.1',
#     '192.168.10.8': 'Win Vista',
#     '192.168.10.14': 'Win 10',
#     '192.168.10.15': 'Win 10',
#     '192.168.10.25': 'macOS'
# }

# Consider all Ubuntu versions as Ubuntu and Windows as Windows
# ip_dict = {
#     '192.168.10.51': 'Ubuntu',
#     '192.168.10.19': 'Ubuntu',
#     '192.168.10.17': 'Ubuntu',
#     '192.168.10.16': 'Ubuntu',
#     '192.168.10.12': 'Ubuntu',
#     '192.168.10.9': 'Windows',
#     '192.168.10.5': 'Windows',
#     '192.168.10.8': 'Windows',
#     '192.168.10.14': 'Windows',
#     '192.168.10.15': 'Windows',
#     '192.168.10.25': 'macOS'
# }

# Consider all Windows as Windows
# ip_dict = {
#     '192.168.10.51': 'Ubuntu server 12',
#     '192.168.10.19': 'Ubuntu 14.4',
#     '192.168.10.17': 'Ubuntu 14.4',
#     '192.168.10.16': 'Ubuntu 16.4',
#     '192.168.10.12': 'Ubuntu 16.4',
#     '192.168.10.9': 'Windows',
#     '192.168.10.5': 'Windows',
#     '192.168.10.8': 'Windows',
#     '192.168.10.14': 'Windows',
#     '192.168.10.15': 'Windows',
#     '192.168.10.25': 'macOS'
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

    # args = option_check()
    # filename = args[0]
    filename = "dataset\\labeled_dataset_r1.csv"
    # df = pd.read_csv(filename)
    # processed_df = data_preprocess(df)
    # labeled_df = add_label(processed_df, ip_dict)
    #
    # print("Total number of packets: ", len(labeled_df))
    # print(labeled_df)
    # labeled_df.to_csv("dataset\\labeled_dataset_r1.csv", encoding='utf-8', index=False)

    filename = "dataset\\labeled_dataset_r1.csv"
    labeled_df = pd.read_csv(filename)

    # fingerprinting with classification

    # x_iloc_list = ['ip.len', 'tcp.window_size',
    #          'tcp.ack', 'tcp.seq', 'tcp.len', 'tcp.stream', 'tcp.analysis.ack_rtt',
    #          'frame.time_relative', 'tcp.time_relative']
    x_iloc_list = [2,7,8,9,10,11,14,15,17]
    y_iloc = 19
    testSize = float(0.2)
    model = ClassifierModel(labeled_df, x_iloc_list, y_iloc, testSize)

    model.ANN()
    model.SVM('linear')
    model.RF()

