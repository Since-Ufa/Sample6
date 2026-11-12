import os

import pandas as pd
import torch

torch.manual_seed(0)
x = torch.rand(2000, 2) * 2 - 1
label = (x[:, 0] ** 2 + x[:, 1] ** 2 < 0.5).int()

os.makedirs("data", exist_ok=True)
pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": label}).to_csv("data/raw.csv", index=False)
print("saved data/raw.csv")
