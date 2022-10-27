import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import p



class Node:
    """
    Class Node
    """
    def __init__(self, value, radius, left = None, right = None, position = None, cl_prob= None):
        self.left = left
        self.data = value
        self.radius = radius
        self.position = position
        self.right = right
        self.prob = cl_prob

def createNode(data, radius, position = None, left = None, right = None, cl_prob = None):
        """
        Utility function to create a node.
        """
        return Node(data, radius, position, left, right, cl_prob)

def height(root):
    # Check if the binary tree is empty
    if root is None:
        return 0 
    # Recursively call height of each node
    leftAns = height(root.left)
    rightAns = height(root.right)
    
    # Return max(leftHeight, rightHeight) at each iteration
    return max(leftAns, rightAns) + 1

def printCurrentLevel(root, level):
    if root is None:
        return
    if level == 1:
        print(root.data, end=" ")
    elif level > 1:
        printCurrentLevel(root.left, level-1)
        printCurrentLevel(root.right, level-1)

def printLevelOrder(root):
    h = height(root)
    for i in range(1, h+1):
       printCurrentLevel(root, i)

def serialize(root):
        
    def post_order(root):
        if root:
            post_order(root.left)
            post_order(root.right)
            ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
        else:
            ret[0] += '#;'           

    ret = ['']
    post_order(root)

    return ret[0][:-1]  # remove last 

#radius = [3, 3, 3]
root = Node(1, [ 23.0585, -69.1120, -49.2585,   1.9735])
root.left = createNode(3, [ 24.2162, -68.5912, -49.0904,   1.9579]) 
root.right = createNode(2, [25.9920, -67.7565, -48.8790,   2.0117]) 
root.right.right = createNode(5, [ 27.7382, -66.9599, -48.5489,   2.0601])
root.right.left = createNode(4, [ 28.7181, -63.3783, -48.0615,   1.1175])
root.right.left.right = createNode(6, [ 28.4217, -63.0804, -47.3805,   0.1139]) 

'''
radius = [2, 2, 2, 2]
root = Node(1, radius)
root.left = createNode(3, radius) 
root.right = createNode(2, radius) 
root.right.right = createNode(5, radius)
root.right.left = createNode(4, radius)
root.right.left.right = createNode(6, radius) 
'''
      
print("arbol")
printLevelOrder(root)
#G = p.arbolAGrafo (root)
plt.figure()
#nx.draw(G, node_size = 150, with_labels = True)
plt.show()

serial = serialize(root)
print("serialized", serial)
#write serialized string to file
file = open("./Trees/test6.dat", "w")
file.write(serial)
file.close() 


root = Node(1, [ -0.189702, -56.7489, -51.1839, 0.8666429417404757])
root.left = createNode(3, [ -0.565011, -56.8315, -50.794, 0.8573367951146668]) 
root.right = createNode(2, [-1.05711, -57.0718, -50.0178, 0.8621060336527253]) 
root.right.right = createNode(8, [-1.1, -58, -52, 0.8621060336527253]) 
root.right.right.right = createNode(12, [-5, -50, -55, 0.8621060336527253]) 
root.right.right.left = createNode(13, [2, --65, --45, 0.8621060336527253]) 
root.right.left = createNode(9, [-0.05711, -55.0718, -48.0178, 0.8621060336527253]) 
root.left.right = createNode(5, [ 8.81779, -54.2887, -56.0856, 1.1151771225759433])
root.left.left = createNode(4, [ -0.565011, -56.8315, -50.794, 0.8573367951146668])
root.left.left.right = createNode(7, [ -0.852245, -56.9181, -50.3749, 0.8430383719817918]) 
root.left.left.left = createNode(6, [ -0.189702, -56.7489, -51.1839, 0.8666429417404757]) 
root.left.left.left.right = createNode(11, [ -1, -58, -49, 0.8666429417404757]) 
root.left.left.left.left = createNode(10, [ 1, -55, -52, 0.8666429417404757]) 
serial = serialize(root)
file = open("./Trees/test7.dat", "w")
file.write(serial)
file.close() 


