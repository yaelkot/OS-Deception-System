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
# ip_dict = {
#      '10.0.0.5': 'Linux',     #was Ubuntu
#      '10.0.0.6': 'Windows',
#      '192.168.0.100': 'Windows',
#      '192.168.1.105': 'Windows',
#      '192.168.1.34': 'Windows',
#      '192.168.1.81': 'MacOS',
#      '192.168.31.59': 'MacOS',
#      '192.168.1.56': 'MacOS',
#      '10.0.0.7': 'Linux',     #was ARCH
#      '10.100.102.8': 'Linux'  #was RHEL
# }

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

ip_dict = {
    #cicids 2017
    '192.168.10.51': 'Ubuntu',
    '192.168.10.19': 'Ubuntu',
    '192.168.10.17': 'Ubuntu',
    '192.168.10.16': 'Ubuntu',
    '192.168.10.12': 'Ubuntu',
    '192.168.10.9': 'Windows',
    '192.168.10.5': 'Windows',
    '192.168.10.8': 'Windows',
    '192.168.10.14': 'Windows',
    '192.168.10.15': 'Windows',
    '192.168.10.25': 'macOS',
    #real traffic
    '10.0.0.6': 'Win 7',
    '192.168.0.100': 'Win10',
    '192.168.1.11': 'Win10',
    '132.73.223.74': 'Win10',
    '192.168.1.34': 'Win11',
    '192.168.1.105': 'Win11',
    '192.168.1.81': 'Mac2017',
    '192.168.31.59': 'Mac2018',
    '192.168.1.56': 'Mac2019',
    '192.168.0.10': 'Mac2020',
    '10.100.102.8': 'Rhel8',
    '192.168.43.80': 'Pop',
    '10.100.102.7': 'Ubuntu2018',
    '10.0.0.7': 'Arch',
    '10.0.0.5': 'Ubuntu 20.4'
}

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

    filename = ".\\real-traffic\\labeled.csv"

    # names = ['ip.flags.df', 'ip.ttl', 'ip.len', 'tcp.srcport',
    #          'tcp.dstport', 'tcp.seq', 'tcp.ack', 'tcp.len',
    #          'tcp.hdr_len', 'tcp.flags.fin', 'tcp.flags.syn',
    #          'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.cwr',
    #          'tcp.window_size', 'tcp.time_delta', 'frame.len',
    #          'average_time_delta',	'std_time_delta', 'average_ttl', 'average_len', 'std_len', 'os']

    names = ['ip.flags.df', 'ip.ttl', 'ip.len', 'tcp.srcport',
                      'tcp.dstport', 'tcp.seq', 'tcp.len',
                      'tcp.hdr_len',  'tcp.flags.push', 'tcp.window_size', 'frame.len',
                      'average_time_delta',	 'average_ttl', 'average_len', 'std_len', 'os']

    labeled_df = pd.read_csv(filename, usecols=names)
    print(labeled_df)

    x_iloc_list = [i for i in range(len(labeled_df.columns)-1)]  # indexes in the labeled csv
    y_iloc = len(labeled_df.columns)-1

    model = ClassifierModel(labeled_df, x_iloc_list, y_iloc)

    filename = "real-traffic\\test.csv"

    test_set = pd.read_csv(filename, usecols=names)

    knn = model.KNN()
    model.SVM('linear')
    model.SVM('rbf')
    model.NB()
    model.RF()
    model.ANN()
    model.DT()

    model.run_models(test_set, x_iloc_list, y_iloc)

    # sklearn.metrics.confusion_matrix(, user_traffic)