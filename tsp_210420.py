import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import namedtuple
from IPython.display import FileLink
import csv
import itertools
import gc


def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def check_solution(solution, points, nodeCount):
    if solution[0] != solution[-1]:
        print("solución inválida, el vértice inicial y el final no son iguales")
        return 0
    else:
        a = solution.pop()
        if len(set(solution)) != len(solution):
            print("solución inválida, existen vértices que se visitan más de una vez")
            return 0
        if len(set(solution)) != len(set(points)):
            print("solución inválida, existen vértices que no se encuentran en el fichero")
            return 0
        else:
            solution.append(a)

            obj = length(points[solution[-1]], points[solution[0]])
            for index in range(0, nodeCount - 1):
                obj += length(points[solution[index]], points[solution[index + 1]])

    return obj


def printGraph(G):
    print(nx.info(G))
    pos = nx.spring_layout(G)
    plt.figure(1)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, with_labels=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()


def greedyAlgorithm(points, nodeCount, taken):
    # points.sort(key=sortPoints)
    cogidos = list(range(0, nodeCount))
    solution = greedy_algorithm(points.copy(), nodeCount, cogidos)

    print(solution)

    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    res = []
    x = 0
    a = points[x]
    while points:
        v = a
        y = x
        weight = float("inf")
        closest = 0
        points.pop(y)
        res.append(taken[y])
        taken.pop(y)
        x = -1
        for u in points:
            if length(v, u) < weight:
                weight = length(v, u)
                closest = u
            x += 1

        a = closest
    return res


# utility function that adds minimum weight matching edges to MST
def minimumWeightedMatching(MST, G, odd_vert):
    while odd_vert:
        v = odd_vert.pop()
        weight = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if G[v][u]['weight'] < weight:
                weight = G[v][u]['weight']
                closest = u
        MST.add_edge(v, closest, weight=weight)
        odd_vert.remove(closest)


def compute(points, nodeCount, i, taken):
    dists = []
    nods = []
    for x in range(0, nodeCount):
        if i != x and taken[x] != 1:
            nods.append(x)
            dists.append(length(points[i], points[x]))
    return dists, nods


def greedy_algorithm_G(G, nodeCount):
    print("greedy")
    res = [0]
    taken = [0]*nodeCount
    taken[0] = 1

    for i in range(0, nodeCount-1):

        nodesF = list(G[i].keys())
        values = [j['weight'] for j in list(G[i].values())]

        nodeWeight = list(zip(nodesF, values))
        nodeWeight.sort(key=lambda x: x[1])

        a = 0
        while taken[nodeWeight[a][0]] == 1:
            a += 1

        res.append(nodeWeight[a][0])
        taken[nodeWeight[a][0]] = 1

    res.append(0)
    return res


def greedy_algorithm_P(points, nodeCount):
    print("greedy")
    res = [0]
    taken = [0] * nodeCount
    taken[0] = 1

    for i in range(0, nodeCount - 1):

        values, nodesF = compute(points, nodeCount, i, taken)

        nodeWeight = list(zip(nodesF, values))
        nodeWeight.sort(key=lambda x: x[1])

        a = 0
        while taken[nodeWeight[a][0]] == 1:
            a += 1

        res.append(nodeWeight[a][0])
        taken[nodeWeight[a][0]] = 1

        clean(values)
        clean(nodesF)
        print(res)

    res.append(0)
    return res


def christofides_algorithm(points, nodeCount):
    """
    1. Create a graph k-complete
    2. Obtain the minimum spanning tree (prim is good for a lot of edges)
    3. Separate nodes with odd degree and get the perfect matching
    4.
    """

    Gr = nx.Graph()

    # Estos for crearán el mismo grafo que crearía la línea de arriba solo que este será con los datos del fichero (k-completo)
    for i in range(0, nodeCount):
        for j in range(0, nodeCount):
            if i != j:
                Gr.add_edge(i, j, weight=length(points[i], points[j]))

    if 2000 < nodeCount < 10000:
        return greedy_algorithm_P(points, nodeCount)

    T = nx.minimum_spanning_tree(Gr, weight='weight', algorithm='prim', ignore_nan=False)

    a = []
    for i in range(0, nodeCount):
        if T.degree[i] % 2 != 0:
            a.append(i)

    eulerM = nx.MultiGraph(T)
    minimumWeightedMatching(eulerM, Gr, a)

    eulerEdges = list(nx.eulerian_circuit(eulerM))
    path = list(itertools.chain.from_iterable(eulerEdges))
    path1 = list(dict.fromkeys(path).keys())
    path1.append(0)

    return path1


def clean(elem):
    del elem
    gc.collect()


def solve_it(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    print(nodeCount)

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    origen = Point(0.0, 0.0)

    def sortPoints(it):
        return length(it, origen)

    points.sort(reverse=True, key=sortPoints)

    #if nodeCount < 2500:
    #    solution = greedy_algorithmP(points, nodeCount)
    #else:
    solution = christofides_algorithm(points, nodeCount)

    obj = check_solution(solution, points, nodeCount)

    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    print("Nodos-> ", nodeCount, " Valor-> ", obj)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    # should call check_solution
    return output_data, obj


if __name__ == "__main__":
    count = 0
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    str_output = [["Filename", "Value"]]
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            full_name = dirname + '/' + filename
            with open(full_name, 'r') as input_data_file:
                input_data = input_data_file.read()
                output, value = solve_it(input_data)
                str_output.append([filename, str(value)])
            count += 1
            print("Proceso->", count, "/76")
    submission_generation('sample_submission_non_sorted.csv', str_output)
    reader = csv.reader(open("sample_submission_non_sorted.csv"))
    sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
    submission_generation('christofides.csv', sortedlist)
