import subprocess
from IPython.display import FileLink
import csv
import os
import math
from collections import namedtuple
import requests
from bs4 import BeautifulSoup

for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def submission_generation(filename, str_output):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return FileLink(filename)


Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def check_solution(solution, points, nodeCount):
    if solution[0] != solution[-1]:
        print("solución inválida, el vértice inicial y el final no son iguales")
        return 0
    else:
        solution.pop()
        if len(set(solution)) != len(solution):
            print("solución inválida, existen vértices que se visitan más de una vez")
            return 0
        elif max(solution) != nodeCount-1 or min(solution) != 0:
            print("Solución inválida, existen vértices que no se encuentran en el fichero")
            return 0
        else:
            obj = length(points[solution[-1]], points[solution[0]])
            for index in range(0, nodeCount - 1):
                obj += length(points[solution[index]], points[solution[index + 1]])

    return obj


def getResult(nodeCount, url):
    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")
    a = soup.findAll('pre')[0].text

    b = a.split('\n')
    cutter = str(nodeCount) + ' ' + str(nodeCount)

    tour = [int(i.split(' ')[0]) for i in b[b.index(cutter)+1:-1]]

    return tour


def getPost(path):
    url_send = 'https://neos-server.org/neos/cgi-bin/nph-neos-solver.cgi'
    info = {'field.4': 'lk',
            'field.5': 'variable',
            'field.6': 'no',
            'solver': 'concorde',
            'inputMethod': 'TSP',
            'auto-fill': 'yes',
            'category': 'co',
            }

    response = requests.post(url_send, data=info, files={'field.1': ('reading', open(path, 'rb'))})

    soup = BeautifulSoup(response.content, "html.parser")

    a = str(soup.findAll('meta')[0])

    indI = a.find('L')+2
    indF = a.find(' ', indI)-1
    return a[indI:indF]


def getConcorde(path, nam):
    run = "./concorde/concorde -x -o " + nam + " -N 2 " + path
    process = subprocess.Popen(run.split(), stdout=subprocess.PIPE)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.kill(process.pid, 0)
        finally:
            return True, None
    output, error = process.communicate()
    if error:
        print("Oh no! ERROR")
        return True, None

    sol = open(nam, "r")
    b = sol.read().split("\n")
    tour = []
    for x in b[1:-1]:
        tour.extend(list(map(int, x.split())))

    return False, tour


def getLin(path, nam):
    run = "./concorde/linkern -o " + nam + " -N 2 -r 2 " + path
    process = subprocess.Popen(run.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Oh no! ERROR")
        return True, None

    sol = open(nam, "r")
    b = sol.read().split('\n')

    tour = [int(i.split(' ')[0]) for i in b[1:-1]]

    return False, tour


"""
Este código sólo ejecuta los dos "scripts" de "Concorde" y "linkern" que se encuentran en la
carpeta ./concorde, y también implementa el resolutor "Lin-kernighan" del servidor web. 
Los ficheros con decimales no fueron calculados aquí, sino en "Kaggle", puesto que estos 
resolutores tienen problemas con ello y no consiguen calcular unos adecuados.
En "Kaggle" se encuentra instalado en el "kernel" el resolutor "Concorde" completo, por lo que
es ahí donde se calcularon estos ficheros (sin embargo, demora mucho tiempo en ellos).
"""

def solve_it(input_data, path):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        a = float(parts[0])
        b = float(parts[1])
        points.append(Point(a, b))

    # El solver de la página web no es capaz de trabajar con decimales,
    # por tanto, hemos separado en dos casos. Llamando en uno al solver
    # de dicha página web, o resolviéndolo con el solver instalado.

    name = "./Temp/" + path[path.index("/", 3)+1:] + ".sol"
    errorB, solution = getConcorde(path, name)

    # Si no se ha podido obtener resultado usando Concorde (errorB=True), se procede a ejecutar "Lin-Kernighan",
    # tanto con el ejecutable "linkern" como con el servidor web, y se obtiene el mejor.
    if errorB:
        errorL, solutionL = getLin(path, name)  # Si la ejecución de "linkern" ha fallado, se devuelve true.

        urlFile = getPost(path) # Se obtiene el resultado del servidor.
        solutionS = getResult(nodeCount, urlFile)

        if errorL: # Si se obtuvo un error ejecutando "linkern", entonces no hace falta compararlos.
            solution = solutionS
        else:
            solutionL.append(solutionL[0])
            objL = check_solution(solutionL, points, nodeCount)

            solutionS.append(solutionS[0])
            objS = check_solution(solutionS, points, nodeCount)

            solution = solutionL if objL <= objS else solutionS  # Se calcula el valor objetivo, y se coge el menor.

    solution.append(solution[0])

    obj = check_solution(solution, points, nodeCount)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data, obj


str_output = [["Filename", "Value"]]
counter = 0
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        full_name = dirname+'/'+filename
        with open(full_name, 'r') as input_data_file:
            input_data = input_data_file.read()
            output, value = solve_it(input_data, full_name)
            str_output.append([filename, str(value)])
        counter += 1
        print(filename, "-->", value, "|| Progreso -> ", counter, "/76")
        print("-----------------------------------------------------------------------------------")

sortedlist = sorted(str_output, key=lambda row: row[0], reverse=False)
submission_generation('tsp_concorde.csv', sortedlist)

