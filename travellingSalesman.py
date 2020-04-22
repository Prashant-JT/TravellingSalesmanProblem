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


def christofides_algorithm(points, nodeCount):
    """
    1. Create a graph k-complete
    2. Obtain the minimum spanning tree (prim is good for a lot of edges)
    3. Separate nodes with odd degree and get the perfect matching
    4. Add nodes and get an Eulerian path
    5. Get Hamiltionan circuit
    """
    Gr = nx.Graph()
    for i in range(0, nodeCount):
        for j in range(0, nodeCount):
            if i != j:
                Gr.add_edge(i, j, weight=length(points[i], points[j]))

    # print("-------------------------")
    # print("Grafo inicial")
    # printGraph(G)
    # print("-------------------------")

    T = nx.minimum_spanning_tree(Gr, weight='weight', algorithm='prim', ignore_nan=False)
    # print("Árbol de expansión mínima ")
    # printGraph(T)
    # print("-------------------------")

    a = []
    for i in range(0, nodeCount):
        if T.degree[i] % 2 != 0:
            a.append(i)

    eulerM = nx.MultiGraph(T)
    minimumWeightedMatching(eulerM, Gr, a)

    eulerEdges = list(nx.eulerian_circuit(eulerM))
    path = list(itertools.chain.from_iterable(eulerEdges))

    # hamiltonian
    path1 = list(dict.fromkeys(path).keys())
    path1.append(0)
    return path1


def solve_it(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    if nodeCount < 1:
        # apply Christofides algorithm
        origen = Point(0.0, 0.0)

        def sortPoints(it):
            return length(it, origen)

        points.sort(reverse=True, key=sortPoints)

        solution = christofides_algorithm(points, nodeCount)
        obj = check_solution(solution, points, nodeCount)
        print("Nodos-> ", nodeCount, " Valor-> ", obj, "(christofides)")

    else:
        min = float('inf')
        solution = [0 for i in range(nodeCount + 1)]
        solution[0] = 0
        visited = [0 for i in range(nodeCount)]
        visited[0] = points[0]
        n = 1

        for i in range(0, nodeCount - 1):
            for j in range(0, nodeCount):
                if points[j] not in visited:
                    if length(visited[n-1], points[j]) < min:
                        min = length(visited[n-1], points[j])
                        min_node = points[j]
                        pos = j
            solution[n] = pos
            visited[n] = min_node
            n += 1
            min = float('inf')

        del(min, min_node, pos, visited, n)
        gc.collect()

        solution[-1] = solution[0]

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
