import p
import networkx as nx
import pickle
import numpy as np
from vec3 import Vec3
import matplotlib.pyplot as plt


filename = "ArteryObjAN1-0"

grafo = pickle.load(open('grafos/' +filename + '-grafo.gpickle', 'rb'))

grafo = grafo.to_undirected()

aRecorrer = []
numeroNodoInicial = 1

distancias = nx.floyd_warshall( grafo )

parMaximo = (-1, -1)
maxima = -1
for nodoInicial in distancias.keys():
    for nodoFinal in distancias[nodoInicial]:
        if distancias[nodoInicial][nodoFinal] > maxima:
            maxima = distancias[nodoInicial] [nodoFinal]
            parMaximo = (nodoInicial, nodoFinal)

for nodo in grafo.nodes:
    if distancias[parMaximo[0]][nodo] == int( maxima / 2):
        numeroNodoInicial = nodo
        break

print(numeroNodoInicial)
rad = grafo.nodes[numeroNodoInicial]['radio']
pos = list(grafo.nodes[numeroNodoInicial]['posicion'].toNumpy())#.append(grafo.nodes[numeroNodoInicial]['radio'])
pos.append(rad)

nodoRaiz = p.Node( numeroNodoInicial, radius =  pos )
for vecino in grafo.neighbors( numeroNodoInicial ):
    if vecino != numeroNodoInicial:
        aRecorrer.append( (vecino, numeroNodoInicial,nodoRaiz ) )

while len(aRecorrer) != 0:
    nodoAAgregar, numeroNodoPadre,nodoPadre = aRecorrer.pop(0)
    posicion = list(grafo.nodes[nodoAAgregar]['posicion'].toNumpy())
    radius = grafo.nodes[nodoAAgregar]['radio']
    posicion.append(radius)
    nodoActual = p.Node( nodoAAgregar, radius =  posicion)
    nodoPadre.agregarHijo( nodoActual )

    for vecino in grafo.neighbors( nodoAAgregar ):
        if vecino != numeroNodoPadre:
            aRecorrer.append( (vecino, nodoAAgregar,nodoActual) )

print(nodoRaiz.children)
print("right", nodoRaiz.left.right.left)
print(nodoRaiz)
serial = nodoRaiz.serialize(nodoRaiz)
print("serialized", nodoRaiz.serialize(nodoRaiz))

#write serialized string to file
file = open("./Trees/ArteryObjAN1-0.dat", "w")
file.write(serial)
file.close() 

def traverse(root, tree):
       
        if root is not None:
            traverse(root.left, tree)
            tree.append((root.radius, root.data))
            traverse(root.right, tree)
            return tree

def traverse_conexiones(root, tree):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            traverse_conexiones(root.left, tree)
            if root.right is not None:
                tree.append((root.data, root.right.data))
            if root.left is not None:
                tree.append((root.data, root.left.data))
            traverse_conexiones(root.right, tree)
            return tree


def arbolAGrafo (nodoRaiz):

    conexiones = []
    lineas = traverse_conexiones(nodoRaiz, conexiones)
    tree = []
    tree = traverse(nodoRaiz, tree)

    vertices = []
    verticesCrudos = []
    for node in tree:
        vertice = node[0][:3]
        rad = node[0][-1]
        num = node[1]
        #vertices.append((num, {'posicion': Vec3( vertice[0], vertice[1], vertice[2]), 'radio': rad} ))
        vertices.append((len(verticesCrudos),{'posicion': Vec3( vertice[0], vertice[1], vertice[2]), 'radio': rad}))
        verticesCrudos.append(vertice)


    G = nx.Graph()
    G.add_nodes_from( vertices )
    G.add_edges_from( lineas )
    
    a = nx.get_node_attributes(G, 'posicion')
    for key in a.keys():
        a[key] = a[key].toNumpy()[0:2]

    #plt.figure(figsize=(20,10))
    #nx.draw(G, pos = a, node_size = 150, with_labels = True)
    #plt.show()
    return G, a

G, a = arbolAGrafo(nodoRaiz)
nx.draw(G, pos = a, node_size = 150, with_labels = True)
plt.show()
import numpy as np
import meshplot as mp
import networkx as nx
mp.offline()
import sys

#from src.mesh_gen import vec3
#sys.modules['vec3'] = vec3
from vec3 import Vec3

grafo = G

posiciones = nx.get_node_attributes( grafo, 'posicion')
p = mp.plot( np.array([ posiciones[node].toNumpy() for node in grafo.nodes]), return_plot=True, shading={'point_size':4})

for arista in grafo.edges:
    p.add_lines( grafo.nodes[arista[0]]['posicion'].toNumpy(), grafo.nodes[arista[1]]['posicion'].toNumpy())