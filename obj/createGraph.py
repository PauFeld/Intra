from parseObj import calcularMatriz, calcularGrafoYArbol
import numpy as np
import networkx as nx
import pickle

filename = 'ArteryObjAN1-10'
from vedo import *
##Load mesh and centerline


#fileObj = open("centerlines/A.obj")
#fileObj = open("centerlines/ArteryObjAN1-0-network")
fileObj = open("centerlines/ " +filename +"-network.obj")
print(fileObj)
grafo = calcularMatriz(fileObj, "radius/" + filename + "-radius.npy")

#nx.number_connected_components(grafo)
#print(grafo)
#nx.write_gpickle(grafo, "grafos/ArteryObjAN1-4-grafo")

with open("grafos/" + filename + '-grafo.gpickle', 'wb') as f:
    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)

