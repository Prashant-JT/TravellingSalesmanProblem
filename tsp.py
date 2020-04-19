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


def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


"""
def check_solution(solution, points, nodeCount, total=0):
    if solution[0] != solution[-1]:
        print("solución inválida, el vértice inicial y el final no son iguales")
        return 0
    else:
        solution.pop()
        if len(set(solution)) != len(solution):
            print("solución inválida, existen vértices que se visitan más de una vez")
            return 0
        if set(solution) != set(points):
            print("solución inválida, existen vértices que no se encuentran en el fichero")
            return 0
        else:
            solution.append(solution[0])
            for i in range(0, nodeCount):
                total += length(solution[i], solution[i + 1])

    return total"""


def printGraph(G):
    print(nx.info(G))
    pos = nx.spring_layout(G)
    plt.figure(1)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, with_labels=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()


def christofides_algorithm(points, nodeCount):
    """
    1. Create a graph k-complete
    2. Obtain the minimum spanning tree (prim is good for a lot of edges)
    3. Separate nodes with odd degree and get the perfect matching
    4.
    """
    matrix = np.zeros(shape=(nodeCount, nodeCount))
    # Estos for crearán el mismo grafo que crearía la línea de arriba solo que este será con los datos del fichero (k-completo)
    for i in range(0, nodeCount):
        for j in range(0, nodeCount):
            matrix[i][j] = length(points[i], points[j])

    G = nx.from_numpy_array(matrix)

    # print("-------------------------")
    # print("Grafo inicial")
    # printGraph(G)
    # print("-------------------------")

    T = nx.minimum_spanning_tree(G, weight='weight', algorithm='prim', ignore_nan=False)
    # print("Árbol de expansión mínima ")
    # printGraph(T)
    # print("-------------------------")

    odds = set()
    for i in range(0, nodeCount):
        if T.degree[i] % 2 != 0:
            odds.add(i)

    a = list(odds)
    odds_1 = np.ix_(a, a)
    odds_2 = nx.from_numpy_array(-1 * matrix[odds_1])
    print("Tiene los impares")
    match = nx.max_weight_matching(odds_2, maxcardinality=True)
    print("Encontró parejas")
    eulerM = nx.MultiGraph(T)

    for edge in match:
        eulerM.add_edge(a[edge[0]], a[edge[1]], weight=matrix[a[edge[0]]][a[edge[1]]])

    a = list(nx.eulerian_circuit(eulerM))
    path = list(itertools.chain.from_iterable(a))
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

    origen = Point(0.0, 0.0)

    def sortPoints(it):
        return length(it, origen)

    points.sort(reverse=True, key=sortPoints)

    solution = christofides_algorithm(points, nodeCount)

    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])

    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    print("Nodos-> ", nodeCount, " Valor-> ", obj)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    # should call check_solution
    return output_data, obj


if __name__ == "__main__":
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
    submission_generation('sample_submission_non_sorted.csv', str_output)
    reader = csv.reader(open("sample_submission_non_sorted.csv"))
    sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
    submission_generation('christofides.csv', sortedlist)
