import os

import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/raw.csv")
df = df.sample(frac=1.0, random_state=params["seed"]).reset_index(drop=True)

split_idx = int(len(df) * (1 - params["test_size"]))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

os.makedirs("data/prepared", exist_ok=True)
train_df.to_csv("data/prepared/train.csv", index=False)
test_df.to_csv("data/prepared/test.csv", index=False)

print(f"prepared train={len(train_df)}, test={len(test_df)}")
