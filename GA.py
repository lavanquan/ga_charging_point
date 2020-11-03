import math
import random
from operator import itemgetter
import copy
import pandas as pd
from ast import literal_eval
from multiprocessing import cpu_count, Process, Pipe
# from pulp import *
import csv
import time
from scipy.stats import sem, t
from scipy import mean


import Network_Frame
for index in range(5):
    Network_Frame.getData("Data/thaydoisonode.csv", index=index)
    print Network_Frame.e
