import numpy as np
import pandas as pd


def data_preprocess(init_df):

    init_df['tcp.srcport'] = np.where(init_df['tcp.srcport'] > 1024, 1, 0)
    init_df['tcp.dstport'] = np.where(init_df['tcp.dstport'] > 1024, 1, 0)

    init_df['ip.dsfield'] = init_df['ip.dsfield'].apply(int, base=16)
    init_df['tcp.flags'] = init_df['tcp.flags'].apply(int, base=16)
    init_df['ip.flags'] = init_df['ip.flags'].apply(int, base=16)

    return init_df


def add_label(df, ip_label_dict):

    # create list of known IP addresses
    known_ip_list = [key for key in ip_label_dict.keys()]
    
    # filter df based on known IP addresses
    all_ip_list = list(set(df['ip.src'].tolist()))
    unnecessary_ip_list = [item for item in all_ip_list if item not in known_ip_list] 
    filtered_df = df.copy()
    for ip in unnecessary_ip_list:
        filtered_df = filtered_df[filtered_df['ip.src'] != ip]

    # print("this is filtered without class~~~~~~~~~~~~~~~~~~~~")
    # print(filtered_df)
    
    # add label to the filtered df
    for ip in known_ip_list:
        filtered_df.loc[filtered_df['ip.src'] == ip, "os"] = ip_label_dict[ip]

    # print(filtered_df)

    return filtered_df
