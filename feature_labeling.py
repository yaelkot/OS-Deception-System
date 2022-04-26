import os
import numpy as np
import pandas as pd

directory = '.\\real-traffic'

ip_dict = {
    '10.0.0.10': 'Win 10',
    '132.73.223.74': 'Win 10',
    '10.100.102.8': 'Rhel 8.5',
    '10.100.102.7': 'Ubuntu 18.04',
    '192.168.1.81': 'Mac 2017',
    '192.168.1.34': 'Win 11',
    '192.168.1.105': 'Win 11',
    '10.0.0.6': 'Win 7',
    '10.0.0.4': 'Win 10',
    '192.168.1.56': 'Mac 2019',
}


def data_preprocess(init_df):
    init_df.dropna()
    init_df['tcp.srcport'] = np.where(init_df['tcp.srcport'] > 1024, 1, 0)
    init_df['tcp.dstport'] = np.where(init_df['tcp.dstport'] > 1024, 1, 0)

    """init_df['ip.dsfield'] = init_df['ip.dsfield'].apply(int, base=16)
    init_df['tcp.flags'] = init_df['tcp.flags'].apply(int, base=16)
    init_df['ip.flags'] = init_df['ip.flags'].apply(int, base=16)"""

    df = init_df.drop(columns=['ip.tos', 'tcp.options.mss_val'])
    return df


def add_label(df, ip_label_dict):
    # create list of known IP addresses
    known_ip_list = [key for key in ip_label_dict.keys()]

    # filter df based on known IP addresses
    all_ip_list = list(set(df['ip.src'].tolist()))
    unnecessary_ip_list = [item for item in all_ip_list if item not in known_ip_list]
    filtered_df = df.copy()
    for ip in unnecessary_ip_list:
        filtered_df = filtered_df[filtered_df['ip.src'] != ip]

    # add label to the filtered df
    for ip in known_ip_list:
        filtered_df.loc[filtered_df['ip.src'] == ip, "os"] = ip_label_dict[ip]

    # print(filtered_df)

    return filtered_df

if __name__ == '__main__':

    for i, filename in enumerate(os.listdir(directory)):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            df = pd.read_csv(directory + '\\' + filename)
            processed_df = data_preprocess(df)
            labeled_df = add_label(processed_df, ip_dict)
            if not i:
                labeled_df.to_csv('.\\labeled.csv', mode='a+', index=False)
            else:
                labeled_df.to_csv('.\\labeled.csv', mode='a+', header=False, index=False)

