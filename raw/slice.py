import numpy as np
import pandas as pd

df = pd.read_csv('techstate_mssa2_new.csv')
df.drop(columns=['Unnamed: 0', 'regimes' ], inplace=True)
overall_len = len(df.index)
validation_len = int(overall_len / 10)
train_len = overall_len - validation_len
df[:train_len].to_csv("train_mssa_dataset.csv")
df[-validation_len:].to_csv("validation_mssa_dataset.csv")
