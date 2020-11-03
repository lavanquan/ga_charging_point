import math
import pandas as pd
from ast import literal_eval


node_pos = []  # location of sensor
charge_pos = []  # location of mc
time_move = []  # time to move
E = []  # energy of sensor
e = []  # energy per second of sensor
numNode = len(node_pos)
numCharge = len(charge_pos)
E_mc = 5  # initial energy of mc
e_mc = 1  # charge per second of mc
E_max = 10.0  # capacity of sensor
e_move = 0.1  # energy per second for move of mc
E_move = [e_move * time_move_i for time_move_i in time_move]  # energy to move of mc
charge_range = 10 ** 10  # charging range of mc
velocity = 0.0  # velocity of mc
alpha = 1
beta = 3
charge = []
depot = (0, 0)


def getData(file_name="data.csv", index=0):
    global node_pos
    global numNode
    global E
    global e
    global charge_pos
    global numCharge
    global time_move
    global E_mc
    global e_mc
    global E_max
    global e_move
    global E_move
    global alpha
    global beta
    global velocity, depot

    df = pd.read_csv(file_name)
    node_pos = list(literal_eval(df.node_pos[index]))
    numNode = len(node_pos)
    E = [df.energy[index] for _ in node_pos]
    e = list(map(float, df.e[index].split(",")))
    charge_pos = list(literal_eval(df.charge_pos[index]))
    numCharge = len(charge_pos)
    velocity = df.velocity[index]
    E_mc = df.E_mc[index]
    E_max = df.E_max[index]
    e_mc = df.e_mc[index]
    e_move = df.e_move[index]
    alpha = df.alpha[index]
    beta = df.beta[index]
    depot = literal_eval(df.depot[index])

    charge_extend = []
    charge_extend.extend(charge_pos)
    charge_extend.append(depot)
    time_move = [[distance(pos1, pos2) / velocity for pos2 in charge_extend] for pos1 in charge_extend]

    tmp = [time_move[i][i + 1] * e_move for i in range(len(time_move) - 1)]
    E_move = [time_move[-1][0] * e_move]
    E_move.extend(tmp)


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) * (node1[0] - node2[0])
                     + (node1[1] - node2[1]) * (node1[1] - node2[1]))


def charging(node, charge_location):
    d = distance(node, charge_location)
    if d > charge_range:
        return 0
    else:
        return alpha / ((d + beta) ** 2)
