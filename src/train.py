import hashlib
import os
import subprocess

import mlflow
import pandas as pd
import torch
import yaml

from model import make_model

params = yaml.safe_load(open("params.yaml"))["train"]

df = pd.read_csv("data/prepared/train.csv")
x = torch.tensor(df[["x1", "x2"]].values, dtype=torch.float32)
y = torch.tensor(df[["label"]].values, dtype=torch.float32)

torch.manual_seed(params["seed"])
model = make_model(params["hidden"])
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
loss_fn = torch.nn.BCEWithLogitsLoss()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("circle-classifier")

with mlflow.start_run(run_name=os.environ.get("DVC_EXP_NAME")) as run:
    mlflow.log_params(params)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    mlflow.set_tag("git_commit", commit)
    mlflow.set_tag("train_data_md5", hashlib.md5(open("data/prepared/train.csv", "rb").read()).hexdigest())
    mlflow.set_tag("dvc_exp", os.environ.get("DVC_EXP_NAME", "none"))

    for epoch in range(params["epochs"]):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    mlflow.log_artifact("models/model.pt")
    with open("models/run_id.txt", "w") as f:
        f.write(run.info.run_id)

print(f"final loss={loss.item():.4f} run_id={run.info.run_id}")
