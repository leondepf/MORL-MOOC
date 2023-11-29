import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

def process_data(lines):
    # 初始化存储数据的列表
    acc_vals = []
    average_time_vals = []
    harmonic_mean_vals = []

    # 正则表达式用于提取每行中的数据
    pattern = r'iteration \d+ : acc_val ([\d\.]+) , average_time_val ([\d\.]+)%, harmonic_mean_val ([\d\.]+)'

    # 遍历每行数据
    for line in lines:
        # 使用正则表达式匹配数据
        match = re.search(pattern, line)
        if match:
            # 将提取的数据转换为浮点数，并截断或补全到小数点后四位
            acc_val = round(float(match.group(1)), 4)
            average_time_val = round(float(match.group(2)), 4)
            harmonic_mean_val = round(float(match.group(3)), 4)

            # 将数据添加到相应的列表
            acc_vals.append(acc_val)
            average_time_vals.append(average_time_val)
            harmonic_mean_vals.append(harmonic_mean_val)

    return acc_vals, average_time_vals, harmonic_mean_vals



if __name__ == '__main__':


    # print(len(ects_acc_vals), len(ects_avg_times) , len(ects_avg_times))
    process_data()
