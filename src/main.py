import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
companies = data.apply(lambda row: {
    'description': row.iloc[0],
    'bussines_tags': row.iloc[1],
    'sector': row.iloc[2],
    'category': row.iloc[3],
    'niche': row.iloc[4]
}, axis=1)


i = 0
x_axis = []
y_axis = []
results = DFT(labels, companies, i)
while i < 2 :
     x_axis.append(i)
     results = {key : value for key, value in results.items() if value[1] > i}
     y_axis.append(len(results))
     i += 0.5

plt.figure(figsize=(8, 6))
plt.plot(x_axis, y_axis, color='orange', linestyle='-', marker='o')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Number of classified items', fontsize=12)
plt.savefig('threshold_plot.pdf', format='pdf')
plt.show()
