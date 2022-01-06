import pandas as pd


def data_preprocess(init_df):

    # df = init_df.drop(columns=['ip.tos', 'tcp.options.mss_val'])

    # df = init_df.loc[:, ['ip.src', 'ip.dst', 'ip.len', 'ip.flags.df', 'ip.flags.mf', 'ip.ttl', 'ip.proto',
    #                      'tcp.window_size', 'tcp.ack', 'tcp.seq', 'tcp.len', 'tcp.stream', 'tcp.urgent_pointer',
    #                      'tcp.flags', 'tcp.analysis.ack_rtt', 'frame.time_relative', 'frame.time_delta',
    #                      'tcp.time_relative', 'tcp.time_delta']]

    df = init_df.loc[:, ['ip.len', 'tcp.window_size',
                         'tcp.ack', 'tcp.seq', 'tcp.len', 'tcp.stream', 'tcp.analysis.ack_rtt',
                         'frame.time_relative', 'tcp.time_relative', 'os']]

    final_df = df.dropna()
    return final_df


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
