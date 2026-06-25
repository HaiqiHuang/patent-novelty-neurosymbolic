import pandas as pd

train = pd.read_csv("/content/drive/MyDrive/IP/data/processed/train.csv")
train.columns = train.columns.str.strip()
train["novelty"] = train["novelty"].astype(int)

small = train.groupby("novelty", group_keys=False).apply(
    lambda x: x.sample(n=100, random_state=42)
).sample(frac=1, random_state=42)

small.to_csv("/content/drive/MyDrive/IP/data/processed/debug_small_train.csv", index=False)
small.to_csv("/content/drive/MyDrive/IP/data/processed/debug_small_valid.csv", index=False)

print(small.shape)
print(small["novelty"].value_counts())