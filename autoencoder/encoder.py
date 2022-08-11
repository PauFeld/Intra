from p import Tree, Node
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import os

def deserialize(data):
        root = None
        tree2 = Tree()
        nodes = data.split(';')
        node = nodes.pop().split(',')
        data = node[0]
        radius = node[1]
        position = node[2]
        r = tree2.insert(root, data, radius, position) 


        def post_order(nodes):
            
            if  not nodes:
                return 
            node = nodes.pop().split(',')
            data = node[0]
            radius = node[1]
            position = node[2]
            tree2.insert(r, data, radius, position)
            root = Node(data, radius, position)
            root.right = post_order(nodes)
            root.left = post_order(nodes)
            tree2.root =r
            return tree2
        return post_order(nodes)    

def read_tree(filename):
    with open('./trees/' +filename, "r") as f:
        byte = f.read() 
        return byte

tree = deserialize(read_tree("ArteryObjAN1-7.dat"))
tree.traverseInorder(tree.root)

print("r", tree.root.radius)



t_list = os.listdir('./trees')

   

class tDataset(Dataset):
    def __init__(self, transform=None):
        self.names = t_list
        self.transform = transform


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        file = t_list[idx]
        string = read_tree(file)
        return string

dataset = tDataset()
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

i = iter(data_loader).next()
print(i)
tree = deserialize(str(i))
tree.traverseInorder(tree.root)




###ENCODER
class Encoder(nn.Module):
    
    def __init__(self, input_size, feature_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        vector = self.encoder(input)
        vector = self.tanh(vector)
        return vector

class bEncoder(nn.Module):
    
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.box_encoder = Encoder(input_size = config.box_code_size, feature_size = config.feature_size)
        
    def Encoder(self, box):
        return self.encoder(box)


def encode_structure_fold(fold, tree):
    
    def encode_node(node):
        
            fold.add('boxEncoder', node)
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('adjEncoder', left, right)

       
        

    encoding = encode_node(tree.root)
    return fold.add('sampleEncoder', encoding)