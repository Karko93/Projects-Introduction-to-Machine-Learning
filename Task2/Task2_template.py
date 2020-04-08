import pandas as pd
import numpy as np

train_features = pd.read_csv("train_features.csv")
test_features = pd.read_csv("test_features.csv")
train_labels = pd.read_csv("train_labels.csv")


### Train set data
X_train_features = train_features.drop(columns=["pid"])
X_train_pid = train_features.get("pid") ## just in case if needed
y_train_labels = train_labels.drop(columns = ["pid"])
train_label_pid = train_labels.get("pid") ## just in case if needed

### Test set data
X_test_features = test_features.drop(columns = ["pid"])
X_test_pid = test_features.get("pid")










###result = pd.concat([X_test_pid,pd.DataFrame(__result here__)],axis=1)
result = pd.concat([X_test_pid.drop_duplicates()],axis=1)

pd.DataFrame(result).to_csv("submit.csv", index=False, header=True)
