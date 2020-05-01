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
from random import sample


def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def generate_random_nodes(nodeCount):
    nodes = [0, 1]
    while abs(nodes[0] - nodes[1]) <= 3:
        nodes = sample(range(0, nodeCount), 2)

    return min(nodes), max(nodes)


def twoOpt(solution, points, nodeCount):
    if nodeCount < 100:
        cont = 10000
    elif nodeCount < 10000:
        cont = 250000
    else:
        cont = 500000

    print("estoy aqui")
    while cont:
        a, b = generate_random_nodes(nodeCount)

        aux = solution[a+1:b]
        distAct = length(points[solution[a]], points[aux[0]]) + length(points[aux[-1]], points[solution[b]])
        aux.reverse()
        distNew = length(points[solution[a]], points[aux[0]]) + length(points[aux[-1]], points[solution[b]])

        if distNew < distAct:
            solution[a+1:b] = aux

        cont -= 1


def check_solution(solution, points, nodeCount):
    if solution[0] != solution[-1]:
        print("solución inválida, el vértice inicial y el final no son iguales")
        return 0
    else:
        a = solution.pop()

        if len(set(solution)) != len(solution):
            print("solución inválida, existen vértices que se visitan más de una vez")
            return 0
        elif max(solution) != nodeCount - 1 or min(solution) != 0:
            print("Solución inválida, existen vértices que no se encuentran en el fichero")
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
        weight = np.inf
        closest = 0
        for u in odds:
            if G[v][u]['weight'] < weight:
                weight = G[v][u]['weight']
                closest = u
        MST.add_edge(v, closest, weight=weight)
        odds.remove(closest)


def christofides_algorithm(points, nodeCount):
    """
    1. Create a graph k-complete
    2. Obtain the minimum spanning tree (prim is good for a lot of edges)
    3. Separate nodes with odd degree and get the perfect matching
    4.
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

    if nodeCount < 2000:
        subGr = nx.Graph()
        for i in range(0, len(a)):
            for j in range(0, len(a)):
                if i != j:
                    subGr.add_edge(i, j, weight=-length(points[a[i]], points[a[j]]))

        match = nx.max_weight_matching(subGr, maxcardinality=True)
        eulerM = nx.MultiGraph(T)

        for edge in match:
            eulerM.add_edge(a[edge[0]], a[edge[1]], weight=Gr[a[edge[0]]][a[edge[1]]]["weight"])
    else:
        eulerM = nx.MultiGraph(T)
        minimumWeightedMatching(eulerM, Gr, a)

    eulerEdges = list(nx.eulerian_circuit(eulerM))
    path = list(itertools.chain.from_iterable(eulerEdges))
    path1 = list(dict.fromkeys(path).keys())

    return path1


def twoInception(kmeans, points, solution, clust):
    subsArr = [0]
    for i in range(0, clust):

        final = list(np.where(kmeans.labels_ == i)[0])

        centroide = Point(kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1])
        final.sort(key=lambda t: length(points[t], centroide))

        v = final.pop()
        a = [v]

        while final:
            weight = np.inf
            closest = 0
            for u in final:
                if length(points[v], points[u]) < weight:
                    weight = length(points[v], points[u])
                    closest = u

            a.append(closest)
            final.remove(closest)
            v = closest

        solution.extend(a)
        subsArr.append(len(solution))

    x = 1
    first = solution[subsArr[0]:subsArr[1]]
    while x < len(subsArr)-2:
        aux = solution[subsArr[x]:subsArr[x + 1]]

        distAct = length(points[first[-1]], points[aux[0]]) + length(points[aux[-1]],
                                                                     points[solution[subsArr[x + 1] + 1]])
        aux.reverse()
        distNew = length(points[first[-1]], points[aux[0]]) + length(points[aux[-1]],
                                                                     points[solution[subsArr[x + 1] + 1]])

        if distNew < distAct:
            solution[subsArr[x]:subsArr[x + 1]] = aux

        first = aux
        x += 1

    return solution


def threeInception(kmeans, points, solution):
    subsArr = [0]
    for i in range(0, 10):

        aux = list(np.where(kmeans.labels_ == i)[0])

        pointsAux = [points[x] for x in aux]

        dictAux = dict(zip(pointsAux, aux))

        arrTwo = pd.DataFrame.from_records(pointsAux)

        kmeansTwo = KMeans(n_clusters=5).fit(arrTwo)

        for j in range(0, 5):
            aux1 = list(np.where(kmeansTwo.labels_ == j)[0])

            final = [dictAux.get(z) for z in [pointsAux[y] for y in aux1]]

            centroide = Point(kmeansTwo.cluster_centers_[j][0], kmeansTwo.cluster_centers_[j][1])
            final.sort(key=lambda t: length(points[t], centroide))

            v = final.pop()
            a = [v]

            while final:
                weight = np.inf
                closest = 0
                for u in final:
                    aux = length(points[v], points[u])
                    if aux < weight:
                        weight = aux
                        closest = u

                a.append(closest)
                final.remove(closest)
                v = closest

            solution.extend(a)
            subsArr.append(len(solution))

    x = 1
    first = solution[subsArr[0]:subsArr[1]]
    while x < len(subsArr)-2:
        aux = solution[subsArr[x]:subsArr[x+1]]

        distAct = length(points[first[-1]], points[aux[0]]) + length(points[aux[-1]], points[solution[subsArr[x+1]+1]])
        aux.reverse()
        distNew = length(points[first[-1]], points[aux[0]]) + length(points[aux[-1]], points[solution[subsArr[x+1]+1]])

        if distNew < distAct:
            solution[subsArr[x]:subsArr[x+1]] = aux

        first = aux
        x += 1

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

        points.sort(reverse=True, key=lambda it: length(it, origen))
        solution = christofides_algorithm(points, nodeCount)
    else:
        arr = pd.DataFrame.from_records(points)

        if nodeCount < 20000:
            cluster = 2
        else:
            cluster = 10

        kmeans = KMeans(n_clusters=cluster).fit(arr)

        solution = []

        if nodeCount < 20000:
            twoInception(kmeans, points, solution, cluster)
        else:
            threeInception(kmeans, points, solution)

    solution.append(solution[0])
    twoOpt(solution, points, nodeCount)
    obj = check_solution(solution, points, nodeCount)

    del points
    gc.collect()

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
    hilos = 0
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
