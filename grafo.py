import numpy as np
import networkx as nx

line = np.load('ArteryObjAN1-0.npy')

adj0 = []
adj1 = []
adj = []
for i in range(line.shape[0]-1):
  point = line[i]
  next_point = line[i+1]
  if point[3] == next_point[3]:
    adj0.append(i)
    adj1.append(i+1)
    adj.append((i, i+1))

print(adj0)
print(adj1)
print(len(adj0))
print(adj)

import igraph
g = igraph.Graph()
g.add_vertices(65)
g.add_edges(adj)
for i in range(65):
  g.vs[i]["coordinates"] = line[i, :3]
  g.vs[i]["radius"] = line[i, 4]

igraph.plot(g)

g.save("ArteryObjAN1-0grafo.graphml")


a = igraph.Graph()
a.Read_Pickle("ArteryObjAN1-0grafo.p")

print(g)
print(a)



G=nx.Graph()
G.add_nodes_from(lpos)
G.add_edges_from(adj)

pos=nx.get_node_attributes(G,'pos')

print(pos)
nx.draw(G,pos)