from createTree import Tree, Node, deserialize
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import os

use_gpu = True

def read_tree(filename):
    with open('./trees/' +filename, "r") as f:
        byte = f.read() 
        return byte

tree = deserialize(read_tree("test.dat"))
tree.traverseInorder(tree.root)
#print("r", tree.root.radius)


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

#i = iter(data_loader).next()
#print(i)
#tree = deserialize(i[0])
#print("deserialized tree")
#tree.traverseInorder(tree.root)



hidden_size = 100
feature_size = 100
###ENCODER


class LeafEncoder(nn.Module):
    
    def __init__(self):
        super(LeafEncoder, self).__init__()
        self.radius_feature = nn.Linear(1, 1)
        #self.position_feature = nn.Linear(3, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        rad = torch.tensor(input.radius)
        
        rad = torch.reshape(rad, (1,1))
        radius = self.radius_feature(rad)
        radius = self.tanh(radius)

        #pos = input.position
        #position = self.position_feature(pos)
        #position = self.tanh(position)
        #feature = torch.cat((radius, position), 1)
        feature = radius
       
        return feature

class NonLeafEncoder(nn.Module):
    
    def __init__(self):
        super(NonLeafEncoder, self).__init__()
        self.radius_feature = nn.Linear(1,1)
        #self.position_feature = nn.Linear(3, 1)
        
        self.left = nn.Linear(1, 1)
        self.right = nn.Linear(1, 1)
        self.encoder = nn.Linear(2, 1)
        self.tanh = nn.Tanh()


    def forward(self, input, left_input, right_input):
        
        radius = self.radius_feature(torch.tensor(input.radius).reshape(1,1))
        #print("rsahpe", radius.shape)
        radius = self.tanh(radius)
        #position = self.position_feature(torch.tensor(input.position))
        #position = self.tanh(position)
        context = self.right(right_input)
        if left_input is not None:
            context += self.left(left_input)
        context = self.tanh(context)
    
        #feature = torch.cat((radius,position,context), 1)
        feature = torch.cat((radius,context), 1)
        feature = self.encoder(feature)
        feature = self.tanh(feature)


        return feature


leafenc = LeafEncoder()
nonleafenc = NonLeafEncoder()


##revisar la recursion
'''
def encode_structure_fold(tree):
    
    def encode_node(node):
        if node.is_leaf():
            return leafenc(node)

        else:
            if node.left is not None:
                left = encode_node(node.left)
            else:
                left = None
            if node.right is not None:
                right = encode_node(node.right)
            else:
                right = None
            return nonleafenc(node, left, right)
        
    encoding = encode_node(tree.root)
    return encoding
'''
def encode_structure_fold(tree):
    c = 0
    def encode_node(node, c):
        
        if node is None:
            return
        print("node: ", node.data, c)
        c+=1
        if node.is_leaf():
            return leafenc(node)

        else:
            left = encode_node(node.left, c)
            right = encode_node(node.right, c)
            return nonleafenc(node, left, right)
        
    encoding = encode_node(tree.root, c)
    return encoding

count = 0
print("counter", tree.count_nodes(tree.root, count))
#device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

#leafencoder = leafenc.to(device)
#nonleafencoder = nonleafenc.to(device)

#print("root",type(tree.root.data))
print(tree.search(tree.root,20).right)
print(tree.search(tree.root,20).left)
p = tree.search(tree.root,18).is_leaf()
print("p", p)
enc_fold_nodes = encode_structure_fold(tree)##voy encodeando nodo por nodo
print("encoding", enc_fold_nodes)


class NodeClassifier(nn.Module):
    
    def __init__(self):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(1, 1)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(1, 3)
        
    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        
        return output

class InternalDecoder(nn.Module):

    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self):
        super(InternalDecoder, self).__init__()
        self.mlp = nn.Linear(1,1)
        self.mlp_right = nn.Linear(1,2)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(1,1)

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        rad_feature = self.mlp2(vector)

        return right_feature, rad_feature

class BifurcationDecoder(nn.Module):
    
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self):
        super(BifurcationDecoder, self).__init__()
        self.mlp = nn.Linear(1,1)
        self.mlp_left = nn.Linear(1,1)
        self.mlp_right = nn.Linear(1,1)
        self.mlp2 = nn.Linear(1,1)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)
        right_feature = self.tanh(right_feature)
        rad_feature = self.mlp2(vector)

        return left_feature, right_feature, rad_feature

class featureDecoder(nn.Module):
    
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self):
        super(featureDecoder, self).__init__()
        self.mlp = nn.Linear(1,1)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(1,1)

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.mlp2(vector)

        return vector

featuredec = featureDecoder()
bifdec = BifurcationDecoder()
internaldec = InternalDecoder()
nodeClassifier = NodeClassifier()


def decode_structure_fold(v):
    
    def decode_node(v):
        cl = nodeClassifier(v)
        _, label = torch.max(cl, 1)
        label = label.data
        #breakpoint ()
        print(label)
        if label == 0: ##output del classifier
            return Node(1, featuredec(v))
        if label == 1:
            right, radius = internaldec(v)
            return Node(1, radius, right = decode_node(right))
        else:
            left, right, radius = bifdec(v)
            return Node(1, radius, decode_node(left), decode_node(right))
           

    dec = decode_node(v)
    return dec


#print(decode_structure_fold(enc_fold_nodes))

'''
for epoch in range(1):
    for data in data_loader:#data es un arbol serializado
        #tree = deserialize(str(data))
        
        #tree = deserialize(data[0])
        
        tree.traverseInorder(tree.root)
        print("root",type(tree.root.data))
        print("d",tree.search(tree.root,18))
        p = tree.search(tree.root,18).is_leaf()
        print("p", p)
        
        
        enc_fold_nodes = encode_structure_fold(tree)##voy encodeando nodo por nodo
        print("encoding", enc_fold_nodes)

            # Apply the computations on the encoder model
      
'''