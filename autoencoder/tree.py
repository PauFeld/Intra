import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import os

filename = "ArteryObjAN1-7"

grafo = pickle.load(open('grafos/' +filename + '-grafo.gpickle', 'rb'))
print(grafo)
grafo = grafo.to_undirected()

adjacency = nx.adjacency_matrix(grafo)

l_nodes = []

l_nodes = [(n, nbrdict) for n, nbrdict in grafo.adjacency()]

class TreeNode:
    
    def __init__(self, tag, radius, posicion):
        self.radius = radius
        self.posicion = posicion
        self.tag = tag
        self.leftChild = None
        self.rightChild=None

class Tree:

    def createNode(self, tag, radius, posicion):
        return TreeNode(tag, radius, posicion)

    def insert(self, node , tag, radius, posicion):
        
        #if tree is empty , return a root node
        if node is None:
            return self.createNode(radius, posicion, tag)
        if tag < node.tag:
            node.leftChild = self.insert(node.leftChild, tag, radius, posicion)
        elif tag > node.tag:
            node.rightChild = self.insert(node.rightChild, tag, radius, posicion)
        
    
    def search(tag, self, node):
       
        # if root is None or root is the search data.
        if node is None or node.tag == tag:
            return node

        if node.tag < tag:
            return self.search(node.rightChild, tag)
        else:
            return self.search(node.leftChild, tag)

    def printTree(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.printTree(root.leftChild)
            print (root.tag, root.radius, root.posicion)
            self.printTree(root.rightChild)

        
#print(l_nodes[0])
#print(l_nodes[0][0])
#print(grafo.nodes[0]['posicion'].toNumpy())

'''
root = None
tree = Tree()
root = tree.insert(root, 10, 5, 1)
#print (root.tag, root.radius)
tree.insert(root, 6, posicion=5, tag=2)
tree.printTree(root)
'''
root = None
tree = Tree()

#agrego el primer nodo
node = l_nodes[0]
tag = node[0]
print(tag)
radius = grafo.nodes[tag]['radio']
posicion = grafo.nodes[tag]['posicion'].toNumpy()
root = tree.insert(root, tag, radius, posicion)

'''
node = l_nodes[1]
tag = node[0]
print(tag)
radius = grafo.nodes[tag]['radio']
posicion = grafo.nodes[tag]['posicion'].toNumpy()
tree.insert(root, tag, radius, posicion)


node = l_nodes[2]
tag = node[0]
print(tag)
radius = grafo.nodes[tag]['radio']
posicion = grafo.nodes[tag]['posicion'].toNumpy()
tree.insert(root, tag, radius, posicion)

node = l_nodes[14]
tag = node[0]
print(tag)
radius = grafo.nodes[tag]['radio']
posicion = grafo.nodes[tag]['posicion'].toNumpy()
tree.insert(root, tag, radius, posicion)
'''
print("////")
for node in l_nodes[1:]:
    tag = node[0]
    radius = grafo.nodes[tag]['radio']
    posicion = grafo.nodes[tag]['posicion'].toNumpy()
    tree.insert(root, tag, radius, posicion)

    
tree.printTree(root)

'''
root = None
tree = Tree()
root = tree.insert(root,10,2,2)
print (root)
tree.insert(root,20,2,2)
print ("Traverse Inorder")
tree.printTree(root)'''

