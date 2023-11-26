import os
import pandas as pd
import numpy as np

### import data
path = "data/train_essays.csv"
absolute_path = os.path.join(os.getcwd(), path)
data = pd.read_csv(absolute_path)

# external data
ex_1 = pd.read_csv(os.path.join(os.getcwd(), "data/train_drcat_04.csv"))[["text", "label", "source"]]
ex_2 = pd.read_csv(os.path.join(os.getcwd(), "data/argugpt.csv"))[["text", "model"]]

data.rename(columns={"generated": "label"}, inplace=True)
data = data[["text", "label"]]
ex_1 = ex_1[["text", "label"]]
ex_2["label"] = 1
ex_2 = ex_2[["text", "label"]]

df = pd.concat([data, ex_1, ex_2])
print(df)
print("##########")
print(df.label.value_counts())
