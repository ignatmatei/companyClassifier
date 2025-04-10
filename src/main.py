import numpy as np
import pandas as pd
import tensorflow as tf
from heuristics import DFT
file = open("../data/labels.txt", "r")
labels = file.readlines()
# eliminate the /t and /n
for i in range(len(labels)):
    labels[i] = labels[i].strip()

# read the csv from ../data/ml_insurance_challenge.csv and convert it to a pandas dataframe
file = open("../data/ml_insurance_challenge.csv", "r")
data = pd.read_csv(file)
# the first column is the description
description = data.iloc[:, 0]
bussines_tags = data.iloc[:, 1]
sector = data.iloc[:, 2]
category = data.iloc[:, 3]
niche = data.iloc[:, 4]

results = DFT(labels, description, bussines_tags, sector, category, niche)
nb = 0
for result in results:
    if result > 0:
        nb += 1

print(nb)
