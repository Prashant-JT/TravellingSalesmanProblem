from random import sample
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import networkx as nx
from collections import namedtuple
from IPython.display import FileLink
import csv
import itertools
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


def twoOpt(solution, pts):
    i = 0
    while i < len(solution) - 4:
        aux = solution[i + 1:i + 4]
        distAct = length(pts[solution[i]], pts[aux[0]]) + length(pts[aux[-1]], pts[solution[i + 4]])
        aux.reverse()
        distNew = length(pts[solution[i]], pts[aux[0]]) + length(pts[aux[-1]], pts[solution[i + 4]])
        if distNew < distAct:
            solution[i + 1:i + 4] = aux
        i += 1
    return solution


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


def twoInception(kmeans, points, solution):
    for i in range(0, 10):
        aux = list(np.where(kmeans.labels_ == i)[0])
        pointsAux = [points[x] for x in aux]
        dictAux = dict(zip(pointsAux, aux))
        arrTwo = pd.DataFrame.from_records(pointsAux)
        kmeansTwo = KMeans(n_clusters=10).fit(arrTwo)

        for j in range(0, 10):
            aux1 = list(np.where(kmeansTwo.labels_ == j)[0])
            final = [dictAux.get(z) for z in [pointsAux[y] for y in aux1]]
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
                centroide = Point(kmeansTwo1.cluster_centers_[k][0], kmeansTwo1.cluster_centers_[k][1])
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

    return solution


def generate_random_nodes(nodeCount):
    nodes = [0, 1]
    # check for adjacency
    while abs(nodes[0] - nodes[1]) <= 2:
        nodes = sample(range(1, nodeCount), 2)

    nodes.sort()
    return nodes[0], nodes[1]


def get_distances(solution, points, nodeCount):
    node1, node2 = generate_random_nodes(nodeCount)
    sub = solution[node1+1:node2]
    actual_distance = length(points[solution[node1]], points[sub[0]]) + length(points[sub[-1]], points[solution[node2]])
    sub.reverse()  # subarray to reverse
    new_distance = length(points[solution[node1]], points[sub[0]]) + length(points[sub[-1]], points[solution[node2]])
    return actual_distance, new_distance, sub, node1, node2


def simulated_annealing(solution, nodeCount, points):
    counter = 0
    delta, alpha = 20, 0.5
    T = (-delta / math.log(0.99, 10))
    N = 10
    n = 0

    while counter <= 2 and T != 0:
        actual_distance, new_distance, sub, node1, node2 = get_distances(solution, points, nodeCount)
        delta = new_distance - actual_distance
        if delta >= 0: # no se mejora
            u = np.random.uniform(0, 1)
            if u < np.exp(-delta / T): # aceptamos el empeoramiento
                print("empeora")
                counter += 1
                solution[node1+1:node2] = sub
        else: # si mejora
            print("mejora")
            counter = 0
            solution[node1+1:node2] = sub
            best_solution = solution.copy()
        T = alpha * T
        print("Temperatura actual : ", T)

    return best_solution


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
        christofides = christofides_algorithm(points, nodeCount)
        christofides.append(christofides[0])
        #solution = twoOpt(christofides, points)
        solution = simulated_annealing(christofides, nodeCount, points)
    else:
        arr = pd.DataFrame.from_records(points)
        kmeans = KMeans(n_clusters=10).fit(arr)
        solution = []
        if nodeCount < 10000:
            twoInception(kmeans, points, solution)
        else:
            threeInception(kmeans, points, solution)

    obj = check_solution(solution, points, nodeCount)
    print("Nodos-> ", nodeCount, " Valor-> ", obj)
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
