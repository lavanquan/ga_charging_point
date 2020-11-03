from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from ast import literal_eval
import csv
import random


f = pd.read_csv("data/thaydoisodiemsac10.csv")
with open("charge_pos.csv", mode="w") as g:
    writer = csv.writer(g)
    for i in range(10):
        for j in range(10):
            x = int(100 * (i + random.random()))
            y = int(100 * (j + random.random()))
            row = [x, y]
            writer.writerow(row)

g.close()

