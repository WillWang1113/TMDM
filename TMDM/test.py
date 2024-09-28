import numpy as np
import pandas as pd
import glob
import os

# datasets = ["electricity", "mfred", "traffic", "ETTh1", "ETTh2", "ETTm1"]
# datasets = ["weather"]
datasets = [
    "electricity",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "exchange_rate",
    # "mfred",
    "traffic",
    "weather"
]
# datasets = ["electricity", "traffic", "ETTh1", "ETTh2", "ETTm1","ETTm2", "exchange_rate", 'mfred']
# datasets = ["electricity", "mfred", "traffic", "ETTh1", "ETTh2", "ETTm1","ETTm2", "exchange_rate", "weather"]
seq_len = {
    "electricity": 96,
    "traffic": 96,
    "ETTh1": 96,
    "ETTh2": 96,
    "ETTm1": 96,
    "ETTm2": 96,
    "exchange_rate": 96,
    "weather": 96,
    # "mfred": 288,
}
pred_len = {
    "electricity": [96, 192, 336, 720],
    "traffic": [96, 192, 336, 720],
    "ETTh1": [96, 192, 336, 720],
    "ETTh2": [96, 192, 336, 720],
    "ETTm1": [96, 192, 336, 720],
    "ETTm2": [96, 192, 336, 720],
    "exchange_rate": [96, 192, 336, 720],
    "weather": [96, 192, 336, 720],
    # "mfred": [288, 432, 576],
}
all_results = []
for d in datasets:
    sl = seq_len[d]
    pls = pred_len[d]
    d_results = []
    for pl in pls:
        print(d, pl)
        m_results = []

        exps = glob.glob(f"results/*{d}*sl{sl}*pl{pl}*")
        exps.sort()

        metrics = np.concatenate(
            [np.load(os.path.join(e, "metrics.npy")).reshape(1, -1) for e in exps]
        )
        metrics = metrics.mean(axis=0)
        m_results.append(metrics)

        df = pd.DataFrame(m_results, columns=["MSE", "CRPS"])
        # df[["MSE", "CRPS"]] = df["metric"].tolist()
        # df = df.drop(columns="metric")
        df["pred_len"] = pl
        df["dataset"] = d
        d_results.append(df)
    d_results = pd.concat(d_results)

    all_results.append(d_results)
df = pd.concat(all_results)
print(df)
df.to_csv('results.csv')
# df = df.drop(columns=["MSE"])
# df = df.reset_index(drop=True)
# df = df.set_index(["dataset", "pred_len", "model"])
# df.stack().unstack(2).to_csv("weatherCRPS.csv")


# print(glob.glob("results/*electricity*DLinear*"))
