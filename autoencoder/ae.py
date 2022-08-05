import pickle
import networkx as nx
import matplotlib.pyplot as plt

filename = "ArteryObjAN1-0"

grafo = pickle.load(open(filename + '-grafo.gpickle', 'rb'))
print(grafo)
nx.draw_networkx(grafo)
