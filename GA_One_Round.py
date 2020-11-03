from GAFrame import *
from multiprocessing import cpu_count, Process, Pipe
import csv
import time
# from scipy import mean
from scipy.stats import sem, t
import numpy as np
import sys


def fitness(person):
    charge_pos = person["location"]
    total_distance = 0
    if len(charge_pos) > 1:
        for i in range(len(charge_pos) - 1):
            total_distance = total_distance + Network_Frame.distance(charge_pos[i], charge_pos[i + 1])
    total_distance = total_distance + Network_Frame.distance(charge_pos[0],
                                                             Network_Frame.depot) + Network_Frame.distance(
        charge_pos[-1], Network_Frame.depot)
    time_move = total_distance / Network_Frame.velocity
    total = 0
    for i in range(len(person["T"]) - 1):
        total = total + time_move + person["T"][i] + sum(person["x"][i])
    x = person["x"][-1]
    total = total + person["T"][-1]
    for u, xu in enumerate(x):
        if xu == 0:
            break
        if u == 0:
            time_to_move = Network_Frame.distance(Network_Frame.depot, charge_pos[u]) / Network_Frame.velocity
        else:
            time_to_move = Network_Frame.distance(charge_pos[u - 1], charge_pos[u]) / Network_Frame.velocity
        total = total + time_to_move + xu
    total = total + person["remain"]
    return total


def repair(person):
    E_mc_now = Network_Frame.E_mc
    E_now = [item for item in Network_Frame.E]
    node_pos = Network_Frame.node_pos
    charge_pos = person["location"]
    total_distance = 0
    if len(charge_pos) > 1:
        for i in range(len(charge_pos) - 1):
            total_distance = total_distance + Network_Frame.distance(charge_pos[i], charge_pos[i + 1])
    total_distance = total_distance + Network_Frame.distance(charge_pos[0],
                                                             Network_Frame.depot) + Network_Frame.distance(
        charge_pos[-1], Network_Frame.depot)
    E_move = total_distance * Network_Frame.e_move / Network_Frame.velocity
    off = {"location": copy.deepcopy(person["location"]), "T": [], "x": [], "remain": -1, "fitness": 0.0}
    isStop = False
    for index, x in enumerate(person["x"]):
        T_max = (Network_Frame.E_max - E_mc_now) / Network_Frame.e_mc
        T = min(T_max, person["T"][index])
        temp_E = [E_now[j] - T * Network_Frame.e[j] for j, _ in enumerate(node_pos)]
        temp_E_mc = E_mc_now + T * Network_Frame.e_mc
        eNode = min(
            [temp_E[j] - Network_Frame.distance(Network_Frame.depot, charge_pos[0]) * Network_Frame.e[
                j] / Network_Frame.velocity for j, _ in enumerate(node_pos)])

        if eNode < 0 or temp_E_mc < E_move:
            off["remain"] = min(
                [E_now[j] / Network_Frame.e[j] for j, _ in enumerate(node_pos) if Network_Frame.e[j] > 0])
            break
        else:
            E_mc_now = temp_E_mc
            E_now = temp_E
            off["T"].append(T)
            row = [0 for u, _ in enumerate(charge_pos)]
            for u, xu in enumerate(x):
                if u == 0:
                    time_move = Network_Frame.distance(Network_Frame.depot, charge_pos[u]) / Network_Frame.velocity
                else:
                    time_move = Network_Frame.distance(charge_pos[u - 1], charge_pos[u]) / Network_Frame.velocity
                eNode = min([E_now[j] - time_move * Network_Frame.e[j] for j, _ in enumerate(node_pos)])
                if eNode < 0:
                    isStop = True
                    break
                p = [min(Network_Frame.charging(node, charge_pos[u]) * xu,
                         Network_Frame.E[j] - E_now[j] + (time_move + xu) * Network_Frame.e[j]) for j, node in
                     enumerate(node_pos)]
                temp_E_mc = E_mc_now - sum(p) - time_move * Network_Frame.e_move
                temp_E = [E_now[j] + p[j] - (time_move + xu) * Network_Frame.e[j] for j, _ in enumerate(node_pos)]

                if min(temp_E) < 0:
                    isStop = True
                    break
                else:
                    row[u] = xu
                    E_mc_now = temp_E_mc
                    E_now = temp_E
            off["x"].append(row)
            if not isStop:
                E_mc_now = E_mc_now - Network_Frame.distance(Network_Frame.depot, charge_pos[
                    -1]) * Network_Frame.e_move / Network_Frame.velocity
                E_now = [E_now[j] - Network_Frame.distance(Network_Frame.depot, charge_pos[-1]) * Network_Frame.e[
                    j] / Network_Frame.velocity for j, _ in
                         enumerate(node_pos)]
            else:
                break
    if off["remain"] == -1:
        off["remain"] = min([E_now[j] / Network_Frame.e[j] for j, _ in enumerate(node_pos) if Network_Frame.e[j] > 0])
    off["fitness"] = fitness(off)
    return off


def individual(p_rand, is_sorted):
    person = {"location": locate_init(is_sorted=is_sorted), "T": [], "x": [], "remain": 0, "fitness": 0}
    r = random.random()
    if r <= p_rand:
        T, x = random_init(E_now=Network_Frame.E, E_mc_now=Network_Frame.E_mc,
                           charge_location=person["location"])
    else:
        T, x = uniform_init(E_now=Network_Frame.E, E_mc_now=Network_Frame.E_mc,
                            charge_location=person["location"])
    person["T"].append(T)
    person["x"].append(x)
    person = repair(person)
    return person


def crossover(father, mother, has_hungary):
    child = {"location": blx_pair(father=father["location"], mother=mother["location"], has_hungary=has_hungary),
             "T": blx_scalar(father=father["T"], mother=mother["T"]),
             "x": [blx_scalar(father=father["x"][i], mother=mother["x"][i]) for i, _ in enumerate(father["x"]) if
                   i < len(mother["x"])],
             "remain": 0, "fitness": 0}
    x_index = len(child["x"])
    while x_index < len(father["x"]):
        child["x"].append(father["x"][x_index])
        x_index = x_index + 1
    child = repair(child)
    return child


def mutation(person, m_rand):
    off = copy.deepcopy(person)
    e = Network_Frame.e
    node_pos = Network_Frame.node_pos
    E_mc_now = Network_Frame.E_mc
    charge_pos = person["location"]
    E_now = [Network_Frame.E[j] for j, _ in enumerate(node_pos)]
    # energy_add = [0 for k, _ in enumerate(node_pos)]
    for k, _ in enumerate(off["x"]):
        E_mc_now = E_mc_now + off["T"][k] * Network_Frame.e_mc
        E_now = [E_now[j] - off["T"][k] * e[j] for j, _ in enumerate(Network_Frame.node_pos)]
        x = person["x"][k]
        for u, xu in enumerate(x):
            if u == 0:
                time = Network_Frame.distance(Network_Frame.depot, charge_pos[u]) / Network_Frame.velocity
            else:
                time = Network_Frame.distance(charge_pos[u - 1], charge_pos[u]) / Network_Frame.velocity
            p = [min(Network_Frame.charging(node, charge_pos[u]) * xu,
                     Network_Frame.E[j] - E_now[j] + (time + xu) * e[j]) for j, node in enumerate(node_pos)]
            E_mc_now = E_mc_now - sum(p) - time * Network_Frame.e_move
            E_now = [E_now[j] + p[j] - (time + xu) * e[j] for j, _ in enumerate(node_pos)]
        E_mc_now -= Network_Frame.distance(Network_Frame.depot,
                                           charge_pos[-1]) * Network_Frame.e_move / Network_Frame.velocity
        E_now = [E_now[j] - Network_Frame.distance(Network_Frame.depot, charge_pos[-1]) * e[j] / Network_Frame.velocity
                 for j, _ in enumerate(node_pos)]

    if min(E_now) < 0 or E_mc_now < 0:
        #  network can not generate a new round
        return -1
    else:
        r = random.random()
        if r <= m_rand:
            tmp = random_init(E_now=E_now, E_mc_now=E_mc_now, charge_location=charge_pos)
        else:
            tmp = uniform_init(E_now=E_now, E_mc_now=E_mc_now, charge_location=charge_pos)

        if tmp != -1:
            T, x = tmp
            off["T"].append(T)
            off["x"].append(x)
            off = repair(off)
            return off
        else:
            #  network can't generate a new round
            return -1


def genetic(start, end, pc, pm, m_random, has_hungary, connection):
    global population
    sub_pop = []
    count = 0
    i = start
    while i < end:
        rc = random.random()
        rm = random.random()
        if rc <= pc:
            j = random.randint(0, population_size - 1)
            while j == i:
                j = random.randint(0, population_size - 1)
            child = crossover(population[i], population[j], has_hungary=has_hungary)
            mutated_child = mutation(child, m_random)
            if mutated_child != -1:
                count += 1
                sub_pop.append(mutated_child)
            else:
                sub_pop.append(child)
        if rm <= pm:
            mutated_child = mutation(population[i], m_random)
            if mutated_child != -1:
                count += 1
                sub_pop.append(mutated_child)
        i += 1
    connection.send([count, sub_pop])
    connection.close()


def evolution(maxIterator, pc, pm, m_random, has_hungary=False, nb_thresh=5):
    global population
    bestFitness = 0.0
    nbIte = 0
    t = 0
    conv = []  # do hoi tu
    while t < maxIterator and nbIte < 500:
        # print "t = ", t, "Fitness = ", population[0]["fitness"]
        count = 0  # dem so lan mutation
        nproc = nb_thresh
        process = []
        connection = []

        for pid in range(nproc):
            connection.append(Pipe())
        for pid in range(nproc):
            pro = Process(target=genetic, args=(10 * pid, 10 * (pid + 1), pc, pm, m_random, has_hungary, connection[pid][1]))
            process.append(pro)
            pro.start()
        for pid in range(nproc):
            nbMutation, sub_pop = connection[pid][0].recv()
            count += nbMutation
            population.extend(sub_pop)
            process[pid].join()
        try:
            population = best_select(population, population_size)
            if population[0]["fitness"] - bestFitness >= 1:
                bestFitness = population[0]["fitness"]
                nbIte = 0
            else:
                nbIte = nbIte + 1

            if t % 100 == 0:
                conv.append((t, population[0]["fitness"]))
        except:
            print(population)
            break
        # max_gen = population[0]["num_gen"]
        # population = selectionBest(population)
        if t % 100 == 0:
            print(t, count, round(population[0]["fitness"], 1))
        t += 1
    # population = selectionBest(population)
    return population[0], conv


#  main task
index = 0
read_name = sys.argv[1]
read_file = "data/" + read_name + ".csv"
write_name = sys.argv[2]
p_random = m_random = float(sys.argv[3])
pm = float(sys.argv[4])
p_sorted = float(sys.argv[5])
has_hungary = float(sys.argv[6])
nb_thresh = int(sys.argv[7])
population_size = 10 * nb_thresh
data_range = int(sys.argv[8])
run_range = int(sys.argv[9])
pc = 0.8
while index < data_range:
    print("Data Set ", index)
    #  open file for write data
    file_name = "ga_one_round/" + write_name + str(index) + ".csv"
    f = open(file_name, mode="w")
    header = ["Lan Chay", "Time", "Co Sac", "Khong Sac"]
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    #  parameter for solve
    sum_lifetime = 0.0
    sum_time = 0.0
    # nbRun = 5
    conv = []  # do hoi tu
    confidence_interval = []
    for idRun in range(run_range):
        random.seed(idRun)
        #  GA running
        Network_Frame.getData(read_file, index=index)
        start_time = time.time()
        population = []
        for _ in range(population_size):
            r = random.random()
            if r < p_sorted:
                population.append(individual(p_rand=p_random, is_sorted=True))
            else:
                population.append(individual(p_rand=p_random, is_sorted=False))
        indi, conv = evolution(maxIterator=5000, pc=pc, pm=pm, m_random=m_random, has_hungary=has_hungary)
        end_time = time.time()
        #  tong hop ket qua
        sum_lifetime = sum_lifetime + indi["fitness"]
        sum_time = sum_time + end_time - start_time
        # write to file
        row = {"Lan Chay": "No." + str(idRun), "Time": end_time - start_time, "Co Sac": indi["fitness"],
               "Khong Sac": min(
                   [Network_Frame.E[j] / Network_Frame.e[j] for j, _ in enumerate(Network_Frame.node_pos) if
                    Network_Frame.e[j] > 0])}
        writer.writerow(row)
        confidence_interval.append(indi["fitness"])

    row = {"Lan Chay": "Average", "Time": sum_time / run_range, "Co Sac": sum_lifetime / run_range,
           "Khong Sac": min([Network_Frame.E[j] / Network_Frame.e[j] for j, _ in enumerate(Network_Frame.node_pos) if
                             Network_Frame.e[j] > 0])}
    writer.writerow(row)
    #  calculate confident interval
    confidence = 0.95
    n = len(confidence_interval)
    m = np.mean(confidence_interval)
    std_err = sem(confidence_interval)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    row = {"Co Sac": h}
    writer.writerow(row)
    f.close()

    print("Done Data Set ", index)
    index = index + 1
print("Done All")
