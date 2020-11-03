import random
from operator import itemgetter
import math
import copy
import Network_Frame
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def ox_cross(father, mother):
    child = []
    cutA = random.randint(1, len(father) - 1)
    cutB = random.randint(1, len(father) - 1)
    while cutB == cutA:
        cutB = random.randint(1, len(father) - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    temp = father[start:end]
    index = 0
    while index < start:
        for item in mother:
            if item in father and item not in temp and item not in child:
                child.append(item)
                break
        index = index + 1
    child.extend(temp)
    index = end
    while index < len(father):
        for item in mother:
            if item in father and item not in child:
                child.append(item)
                break
        index = index + 1
    index = len(child)
    while index < len(father):
        for item in father:
            if item not in child:
                child.append(item)
                break
        index = index + 1
    return child


def blx_scalar(father, mother):
    second_gen = copy.copy(mother)
    n = len(second_gen)
    if n < len(father):
        second_gen.extend(father[n:])
    child = []
    for i, _ in enumerate(father):
        min_gen = max(0, min(father[i], second_gen[i]) - 0.5 * math.fabs(father[i] - second_gen[i]))
        max_gen = min(1000, max(father[i], second_gen[i]) + 0.5 * math.fabs(father[i] - second_gen[i]))
        gen = random.random() * (max_gen - min_gen) + min_gen
        child.append(gen)
    return child


def blx_pair(father, mother, has_hungary=1.0):
    second_gen = copy.copy(mother)
    if random.random() < has_hungary:
        second_gen = hungary(father, second_gen)
    n = len(second_gen)
    if n < len(father):
        second_gen.extend(father[n:])
    child = []
    for i, _ in enumerate(father):
        min_gen_x = max(0, min(father[i][0], second_gen[i][0]) - 0.5 * math.fabs(father[i][0] - second_gen[i][0]))
        max_gen_x = min(1000, max(father[i][0], second_gen[i][0]) + 0.5 * math.fabs(father[i][0] - second_gen[i][0]))
        gen_x = random.random() * (max_gen_x - min_gen_x) + min_gen_x
        min_gen_y = max(0, min(father[i][1], second_gen[i][1]) - 0.5 * math.fabs(father[i][1] - second_gen[i][1]))
        max_gen_y = min(1000, max(father[i][1], second_gen[i][1]) + 0.5 * math.fabs(father[i][1] - second_gen[i][1]))
        gen_y = random.random() * (max_gen_y - min_gen_y) + min_gen_y
        child.append((gen_x, gen_y))
    return child


def gauss_cross(father, mother):
    return 0


def gauss_mutate():
    return 0


def swap_mutate():
    return 0


def best_select(population, population_size):
    new_list = sorted(population, key=itemgetter("fitness"), reverse=True)
    return new_list[:population_size]


def random_init(E_now, E_mc_now, charge_location):
    node_pos = Network_Frame.node_pos
    e = Network_Frame.e
    total_distance = 0
    if len(charge_location) > 1:
        for i in range(len(charge_location) - 1):
            total_distance = total_distance + Network_Frame.distance(charge_location[i], charge_location[i + 1])
    total_distance = total_distance + Network_Frame.distance(charge_location[0],
                                                             Network_Frame.depot) + Network_Frame.distance(
        charge_location[-1], Network_Frame.depot)
    E_move = total_distance * Network_Frame.e_move / Network_Frame.velocity
    x = [0 for charge_index, _ in enumerate(charge_location)]
    # array store energy of sensor at new route
    eNode = [E_now[node_index] / e[node_index] for node_index, _ in enumerate(node_pos) if e[node_index] > 0]
    T_max = min(min(eNode), (Network_Frame.E_max - E_mc_now) / Network_Frame.e_mc)
    T_min = max(0, (E_move - E_mc_now) / Network_Frame.e_mc)

    if T_max >= T_min:
        # value of T
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        # Energy of MC and sensor before u location
        E_mc_new = E_mc_now + T * Network_Frame.e_mc - E_move

        energy_receive = [sum([Network_Frame.charging(node, charge) for node_index, node in enumerate(node_pos)]) for
                          charge_index, charge in
                          enumerate(charge_location)]
        p = [charge_index for charge_index, _ in enumerate(charge_location)]
        random.shuffle(p)
        for charge_index in p:
            x[charge_index] = random.random() * E_mc_new / energy_receive[charge_index]
            E_mc_new -= energy_receive[charge_index] * x[charge_index]
        return [T, x]
    else:
        return -1


def uniform_init(E_now, E_mc_now, charge_location):
    node_pos = Network_Frame.node_pos
    e = Network_Frame.e
    total_distance = 0
    if len(charge_location) > 1:
        for i in range(len(charge_location) - 1):
            total_distance = total_distance + Network_Frame.distance(charge_location[i], charge_location[i + 1])
    total_distance = total_distance + Network_Frame.distance(charge_location[0],
                                                             Network_Frame.depot) + Network_Frame.distance(
        charge_location[-1], Network_Frame.depot)
    E_move = total_distance * Network_Frame.e_move / Network_Frame.velocity
    x = [0 for charge_index, _ in enumerate(charge_location)]
    # array store energy of sensor at new route
    eNode = [E_now[node_index] / e[node_index] for node_index, _ in enumerate(node_pos) if e[node_index] > 0]
    T_max = min(min(eNode), (Network_Frame.E_max - E_mc_now) / Network_Frame.e_mc)
    T_min = max(0, (E_move - E_mc_now) / Network_Frame.e_mc)

    if T_max >= T_min:
        # value of T
        T = T_max - 0.2 * abs(T_max - T_min) * random.random()
        # Energy of MC and sensor before u location
        E_mc_new = E_mc_now + T * Network_Frame.e_mc - E_move

        energy_receive = [sum([Network_Frame.charging(node, charge) for node_index, node in enumerate(node_pos)]) for
                          charge_index, charge in
                          enumerate(charge_location)]
        tmp = (E_mc_new - E_move) / sum(energy_receive)
        for charge_index, _ in enumerate(charge_location):
            x[charge_index] = tmp + 0.5 * tmp * (2 * random.random() - 1)
        return [T, x]
    else:
        return -1


def locate_init(is_sorted):
    location = Network_Frame.node_pos
    e = Network_Frame.e
    node = [{"e": e[i], "location": location[i]} for i, _ in enumerate(location)]
    if is_sorted:
        node = sorted(node, key=itemgetter("e"), reverse=True)
    charge_loc = []
    while node:
        if node[0]["e"] == 0:
            node.remove(node[0])
            continue
        d = math.sqrt(Network_Frame.alpha / node[0]["e"]) - Network_Frame.beta
        if d < 0:
            d = 0
        first_loc = node[0]["location"]
        x_min = max(0, first_loc[0] - d)
        x_max = min(1000, first_loc[0] + d)
        x = random.random() * (x_max - x_min) + x_min
        y_min = max(0, first_loc[1] - math.sqrt(d ** 2 - (x - first_loc[0]) ** 2))
        y_max = min(1000, first_loc[1] + math.sqrt(d ** 2 - (x - first_loc[0]) ** 2))
        y = random.random() * (y_max - y_min) + y_min
        charge_loc.append((x, y))
        #  check total node is covered by (x, y)
        new_node = []
        for item in node:
            d = math.sqrt((x - item["location"][0]) ** 2 + (y - item["location"][1]) ** 2)
            if Network_Frame.alpha / (d + Network_Frame.beta) ** 2 <= item["e"] and d >= 10 ** -3:
                new_node.append(item)
        node = new_node
    return charge_loc


def hungary(first, second):
    cost = cdist(first, second)
    row_ind, col_ind = linear_sum_assignment(cost)
    temp = first.copy()
    for index, _ in enumerate(col_ind):
        temp[row_ind[index]] = second[col_ind[index]]
    return temp


#  main task
# Network_Frame.getData("Data/thaydoisonode.csv", index=0)
# charge_pos = locate_init(is_sorted=True)
# print uniform_init(E_now=Network_Frame.E, E_mc_now=Network_Frame.E_mc, charge_location=charge_pos)
# print(hungary([(3, 3), (4, 4), (2, 2), (1, 1)], [(1, 3), (3, 1), (2, 3)]))
