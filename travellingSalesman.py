import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import namedtuple
from IPython.display import FileLink
import csv

def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def check_solution(obj):
    # comprobar que todos los nodos son distintos (all different), recorrer la lista
    # sumando los pesos y que acabe en la misma ciudad en la que empezó
    if True:
        return obj
    return null

def printGraph(g):
    print(nx.info(g))
    nx.draw(g, with_labels=True)
    plt.show()

def solve_it(input_data):
    """Christofides algortithm"""
    lines = input_data.split('\n')
    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    # create a graph k-complete
    G = nx.complete_graph(nodeCount, create_using=nx.Graph())
    printGraph(G)

    # obtain the minimum spanning tree (prim is good for a lot of edges)
    T = nx.minimum_spanning_tree(G, weight='None', algorithm='prim', ignore_nan=False)
    printGraph(T)

    # seperate nodes with odd degree and get perfect matching
    W = {}
    for i in range(0, nodeCount - 1):
        print(i, T.degree[i])
        if T.degree[i] % 2 != 0:
            W[i] = T.degree[i]
    #M = nx.is_perfect_matching(G, matching=W)
    #print(W)

    return output_data, check_solution(obj)

if __name__ == "__main__":
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    str_output = [["Filename","Value"]]
    for dirname, _, filenames in os.walk('data'):
        for filename in filenames:
            full_name = dirname+'/'+filename
            with open(full_name, 'r') as input_data_file:
                input_data = input_data_file.read()
                output, value = solve_it(input_data)
                str_output.append([filename, str(value)])
    submission_generation('sample_submission_non_sorted.csv', str_output)
    reader = csv.reader(open("sample_submission_non_sorted.csv"))
    sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
    submission_generation('christofides.csv', sortedlist)