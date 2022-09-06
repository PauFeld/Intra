import p
import networkx as nx
import pickle

filename = "ArteryObjAN1-2"

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
file = open("./Trees/ArteryObjAN1-2.dat", "w")
file.write(serial)
file.close() 