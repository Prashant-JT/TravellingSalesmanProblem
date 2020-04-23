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
from sklearn.cluster import KMeans

def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def gc_collect(*param):
    for i in param:
        del i
    gc.collect()


def check_solution(solution, points, nodeCount):
    if solution[0] != solution[-1]:
        print("solución inválida, el vértice inicial y el final no son iguales")
        return 0
    else:
        a = solution.pop()
        if len(set(solution)) != len(solution):
            print("solución inválida, existen vértices que se visitan más de una vez")
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


def minimumWeightedMatching(MST, G, odds):
    while odds:
        v = odds.pop()
        weight = float("inf")
        closest = 0
        for u in odds:
            if G[v][u]['weight'] < weight:
                weight = G[v][u]['weight']
                closest = u
        MST.add_edge(v, closest, weight=weight)
        odds.remove(closest)


def christofides_algorithm(points, nodeCount):
    """
    1. Create a k-complete graph
    2. Get the minimum spanning tree
    3. Get nodes with odd degree and get the perfect matching
    4. Add nodes and get an Eulerian path
    5. Get a Hamiltonian cycle
    """
    Gr = nx.Graph()
    for i in range(0, nodeCount):
        for j in range(0, nodeCount):
            if i != j:
                Gr.add_edge(i, j, weight=length(points[i], points[j]))

    T = nx.minimum_spanning_tree(Gr, weight='weight', algorithm='prim', ignore_nan=False)

    a = []
    for i in range(0, nodeCount):
        if T.degree[i] % 2 != 0:
            a.append(i)

    eulerM = nx.MultiGraph(T)
    gc_collect(T)
    minimumWeightedMatching(eulerM, Gr, a)
    gc_collect(Gr, a)

    # euler
    eulerEdges = list(nx.eulerian_circuit(eulerM))
    gc_collect(eulerM)
    path = list(itertools.chain.from_iterable(eulerEdges))
    gc_collect(eulerEdges)

    # hamilton
    hamiltonian_cycle = list(dict.fromkeys(path).keys())
    gc_collect(path)
    hamiltonian_cycle.append(0)
    return hamiltonian_cycle


def twoInception(kmeans, points, solution):
    for i in range(0, 10):
        aux = list(np.where(kmeans.labels_ == i)[0])
        pointsAux = [points[x] for x in aux]
        dictAux = dict(zip(pointsAux, aux))
        arrTwo = pd.DataFrame.from_records(pointsAux)
        kmeansTwo = KMeans(n_clusters=10).fit(arrTwo)
        for j in range(0, 10):
            aux = list(np.where(kmeansTwo.labels_ == j)[0])
            final = [dictAux.get(z) for z in [pointsAux[y] for y in aux]]
            solution.extend(final)
    return solution


def threeInception(kmeans, points, solution):
    for i in range(0, 10):
        aux = list(np.where(kmeans.labels_ == i)[0])
        pointsAux = [points[x] for x in aux]
        dictAux = dict(zip(pointsAux, aux))
        arrTwo = pd.DataFrame.from_records(pointsAux)
        kmeansTwo = KMeans(n_clusters=10).fit(arrTwo)
        for j in range(0, 10):
            aux1 = list(np.where(kmeansTwo.labels_ == j)[0])
            pointsAux1 = [pointsAux[x] for x in aux1]
            dictAux1 = dict(zip(pointsAux1, aux1))
            arrTwo1 = pd.DataFrame.from_records(pointsAux1)
            kmeansTwo1 = KMeans(n_clusters=10).fit(arrTwo1)
            for k in range(0, 10):
                a = list(np.where(kmeansTwo1.labels_ == k)[0])
                final1 = [dictAux1.get(z) for z in [pointsAux1[y] for y in a]]
                final = [dictAux.get(z) for z in [pointsAux[y] for y in final1]]
                solution.extend(final)
    return solution


def solve_it(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    if nodeCount < 4000:
         origen = Point(0.0, 0.0)

        def sortPoints(it):
            return length(it, origen)

        points.sort(reverse=True, key=lambda it: length(it, origen))
        solution = christofides_algorithm(points, nodeCount)
    else:
        arr = pd.DataFrame.from_records(points)

        kmeans = KMeans(n_clusters=10).fit(arr)
        solution = []

        if nodeCount < 10000:
            twoInception(kmeans, points, solution)
        else:
            threeInception(kmeans, points, solution)

        solution.append(solution[0])
        
    obj = check_solution(solution, points, nodeCount)    
    print("Nodos-> ", nodeCount, " Valor-> ", obj, "(greedy)")

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data, obj


if __name__ == "__main__":
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    str_output = [["Filename", "Value"]]
    counter = 0
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            full_name = dirname + '/' + filename
            with open(full_name, 'r') as input_data_file:
                input_data = input_data_file.read()
                output, value = solve_it(input_data)
                str_output.append([filename, str(value)])
            counter += 1
            print("Progreso-> ", counter, "/76")
    submission_generation('sample_submission_non_sorted.csv', str_output)
    reader = csv.reader(open("sample_submission_non_sorted.csv"))
    sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
    submission_generation('christofides.csv', sortedlist)
