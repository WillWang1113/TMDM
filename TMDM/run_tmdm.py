import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# RUN TRAINING

seq_len = 96
# pred_len = [720]
pred_len = [720]
root_dir = "/home/user/data/THU-timeseries"
data_dir = (
    # "ETT-small/ETTh1.csv",
    # "ETT-small/ETTh2.csv",
    # "exchange_rate/exchange_rate.csv",
    # "electricity/electricity.csv",
    "traffic/traffic.csv",
    "weather/weather.csv",
    "ETT-small/ETTm1.csv",
    "ETT-small/ETTm2.csv"
)

for pl in pred_len:
    for d in data_dir:
        if d.split("/")[-1][:-4] not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
            data_class = "custom"
        else:
            data_class = d.split("/")[-1][:-4]

        model_id = d.split("/")[-1][:-4] + f"_{seq_len}_{pl}"
        os.system(
            f"python runner9_NS_transformer.py --model_id {model_id} --data {data_class} --pred_len {pl} --data_path {d} --gpu 1"
        )

# seq_len = 288
# pred_len = [288, 432, 576]
# root_dir = "/home/user/data/THU-timeseries"
# data_dir = ["mfred"]

# for pl in pred_len:
#     for d in data_dir:
#         os.system(
#             f"python iclr24_main_default.py --seq_len {seq_len} --pred_len {pl} --label_len {seq_len} --target -1 --features S --data {d}"
#         )
