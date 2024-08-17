import os.path as osp
import configparser
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#转换时间戳
def convert_seconds(seconds):

    hours = int(seconds // 3600)
    remainder_min = seconds % 3600
    minutes = int(remainder_min // 60)
    remainder_second = seconds % 60
    second = round(remainder_second,0)
    if hours:
        time_str = str(hours) + "小时" + str(minutes) + "分钟" + str(second) + "秒"
    elif minutes:
        time_str = str(minutes) + "分钟" + str(second) + "秒"
    else:
        time_str = str(second) + "秒"

    return time_str


def data2xlsx(data, path):
    for i in data.keys():
        data[i] = [data[i]]
    df = pd.DataFrame(data)
    if os.path.exists(path=path):
        old_data = pd.read_excel(path)
        new_data = old_data._append(df)
        new_data.to_excel(path, index=False)
    else:
        df.to_excel(path, index=False)


def result_file(conf, acc_dict, num_exams, use_time):
    path = osp.join(osp.dirname(osp.abspath(__file__)), "result", conf.dataset_name)
    if not osp.exists(path):
        os.makedirs(path)

    result_xlsx = osp.join(path, conf.model + '__result.xlsx')
    result = {}

    result["序号"] = num_exams

    t = time.localtime()

    conf_dict = conf.__dict__
    for i in conf_dict.keys():
        result[i] = conf_dict[i]
    for i in acc_dict.keys():
        result[i] = acc_dict[i]

    result["运行完成时间"] = str(t.tm_mon) + "月" + str(t.tm_mday) + "日\t" + str(t.tm_hour) + "时" + str(t.tm_min) + "分"
    result["消耗时间"] = convert_seconds(use_time)

    data2xlsx(result, result_xlsx)










