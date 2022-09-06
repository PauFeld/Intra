from logging import raiseExceptions
from tokenize import Double
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


def count_fn(f):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return f(*args, **kwargs)
    wrapper.count = 0
    return wrapper

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
        self.children = [self.left, self.right]
    
    def agregarHijo(self, children):

        if self.right is None:
            self.right = children
        elif self.left is None:
            self.left = children

        else:
            raise ValueError ("solo arbol binario ")


    def is_leaf(self):
        if self.right is None:
            return True
        else:
            return False

    def is_two_child(self):
        if self.right is not None and self.left is not None:
            return True
        else:
            return False

    def is_one_child(self):
        if self.is_two_child():
            return False
        elif self.is_leaf():
            return False
        else:
            return True

    def childs(self):
        if self.is_leaf():
            return 0
        if self.is_one_child():
            return 1
        else:
            return 2
    
    
    def traverseInorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorder(root.left)
            print (root.data, root.radius)
            self.traverseInorder(root.right)

    def traverseInorderLoss(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderLoss(root.left, loss)
            loss.append(root.prob)
            self.traverseInorderLoss(root.right, loss)
            return loss

    def preorder(self, root):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            print (root.data, root.radius)
            self.preorder(root.left)
            self.preorder(root.right)

    def cloneBinaryTree(self, root):
     
        # base case
        if root is None:
            return None
    
        # create a new node with the same data as the root node
        root_copy = Node(root.data, root.radius)
    
        # clone the left and right subtree
        root_copy.left = self.cloneBinaryTree(root.left)
        root_copy.right = self.cloneBinaryTree(root.right)
    
        # return cloned root node
        return root_copy

    def cloneWithoutZero(self, root):
         
        # base case
        if root is None:
            return None
    
        # create a new node with the same data as the root node
        if root.data != 0:
            root_copy = Node(root.data, root.radius)
    
            #    clone the left and right subtree
            root_copy.left = self.cloneBinaryTree(root.left)
            root_copy.right = self.cloneBinaryTree(root.right)
    
            # return cloned root node
            return root_copy

    def height(self, root):
    # Check if the binary tree is empty
        if root is None:
            return 0 
        # Recursively call height of each node
        leftAns = self.height(root.left)
        rightAns = self.height(root.right)
    
        # Return max(leftHeight, rightHeight) at each iteration
        return max(leftAns, rightAns) + 1

    # Print nodes at a current level
    def printCurrentLevel(self, root, level):
        if root is None:
            return
        if level == 1:
            print(root.data, end=" ")
        elif level > 1:
            self.printCurrentLevel(root.left, level-1)
            self.printCurrentLevel(root.right, level-1)

    def printLevelOrder(self, root):
        h = self.height(root)
        for i in range(1, h+1):
            self.printCurrentLevel(root, i)

    
    def CurrentLevel(self, root, level, l, flag):
        
        if root is None:
            return 
        if level == 1:
            #print(root.data)
            if flag == 2:
                if root.data == 0:
                    l.append('#')
                else:
                    l.append(root.radius)
            
            else:
                if root.data == 0 and root.prob is None:
                    l.append('#')
                else:
                    if flag == 0: #number of child
                        l.append(root.childs())
                    if flag == 1: #probability
                        l.append(root.prob)
                    
            return l

        elif level > 1:
            if root.left is None:
                root.left = Node(0,0)
            if root.right is None:
                root.right = Node(0,0)
            self.CurrentLevel(root.left, level-1, l, flag)
            self.CurrentLevel(root.right, level-1, l, flag)

    def LevelOrder(self, root, flag):
        h = self.height(root)
        l = []
        for i in range(1, h+1):
            self.CurrentLevel(root, i, l, flag)
        return l

    
    def count_nodes(self, root, counter):
        
        if   root is not None:
            self.count_nodes(root.left, counter)
            counter.append(root.data)
            self.count_nodes(root.right, counter)
            return counter

    
    def serialize(self, root):
        
        def post_order(root):
            if root:
                post_order(root.left)
                post_order(root.right)
                ret[0] += str(root.data)+'_'+ str(root.radius) +';'
                
            else:
                ret[0] += '#;'           

        ret = ['']
        post_order(root)

        return ret[0][:-1]  # remove last ,

    def search(self, node, data):
        """
        Search function will search a node into tree.
        """
        # if root is None or root is the search data.
        if node is None or node.data == data:
            return node
        if node.data < data:
            return self.search(node.right, data)
        else:
            return self.search(node.left, data)


@count_fn
def createNode(data, radius, position = None, left = None, right = None, cl_prob = None):
        """
        Utility function to create a node.
        """
        return Node(data, radius, position, left, right, cl_prob)
 
def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    #print("node",nodes[3])
    def post_order(nodes):
                
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop().split('_')
        data = int(node[0])
        #radius = float(node[1])
        #print("node", node)
        #breakpoint()
        radius = node[1]
        #print("radius", radius)
        '''
        rad = radius.split(",")
        rad [0] = rad[0].replace('[','')
        rad [3] = rad[3].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        r = torch.tensor(r, device=device)
        '''
        r =[float(num) for num in radius if num.isdigit()]
        r = torch.tensor(r, device=device)
        #breakpoint()
        root = createNode(data, r)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)    


def read_tree(filename):
    with open('./trees/' +filename, "r") as f:
        byte = f.read() 
        return byte

###ENCODER
class LeafEncoder(nn.Module):
    
    def __init__(self):
        super(LeafEncoder, self).__init__()
        self.radius_feature = nn.Linear(3, 32)
        self.tanh = nn.Tanh()

    def forward(self, input):
        rad = torch.tensor(input.radius)
        rad = torch.reshape(rad, (1,3)).to(device)
        radius = self.radius_feature(rad)
        radius = self.tanh(radius)
        feature = radius
       
        return feature

class NonLeafEncoder(nn.Module):
    
    def __init__(self):
        super(NonLeafEncoder, self).__init__()
        self.radius_feature = nn.Linear(3,32)
        self.left = nn.Linear(32, 32)
        self.right = nn.Linear(32, 32)
        self.encoder = nn.Linear(64, 32)
        self.tanh = nn.Tanh()


    def forward(self, input, left_input, right_input):
        
        radius = self.radius_feature(torch.tensor(input.radius).reshape(1,3).to(device))
        radius = self.tanh(radius)
        context = self.right(right_input)
        if left_input is not None:
            context += self.left(left_input)
        context = self.tanh(context)
    
        feature = torch.cat((radius,context), 1)
        feature = self.encoder(feature)
        feature = self.tanh(feature)


        return feature

leafenc = LeafEncoder()
nonleafenc = NonLeafEncoder()
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print("device:", device)
leafenc = leafenc.to(device)
nonleafenc = nonleafenc.to(device)
def encode_structure_fold(root):

    def encode_node(node):
        
        if node is None:
            return
        if node.is_leaf():
            return leafenc(node)
        else:
            left = encode_node(node.left)
            right = encode_node(node.right)
            return nonleafenc(node, left, right)
        
    encoding = encode_node(root)
    return encoding


class NodeClassifier(nn.Module):
    
    def __init__(self):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(32, 8)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(8, 3)
        
    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        
        return output

class InternalDecoder(nn.Module):

    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self):
        super(InternalDecoder, self).__init__()
        self.mlp = nn.Linear(32,16)
        self.mlp_right = nn.Linear(16,32)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(16,3)

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
        self.mlp = nn.Linear(32,32)
        self.mlp_left = nn.Linear(32,32)
        self.mlp_right = nn.Linear(32,32)
        self.mlp2 = nn.Linear(32,3)
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
        self.mlp = nn.Linear(32,16)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(16,3)

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.mlp2(vector)

        return vector

featuredec = featureDecoder()
featuredec=featuredec.to(device)
bifdec = BifurcationDecoder()
bifdec = bifdec.to(device)
internaldec = InternalDecoder()
internaldec=internaldec.to(device)
nodeClassifier = NodeClassifier()
nodeClassifier = nodeClassifier.to(device)

def calcularLossEstructura(cl_p, original):
    if original is None:
        return 0
    else:
        if original.childs() == 0:
            vector = [1, 0, 0]
        if original.childs() == 1:
            vector = [0, 1, 0]
        if original.childs() == 2:
            vector = [0, 0, 1]
    #breakpoint()
    l2 = nn.MSELoss()
    return l2(torch.tensor(vector, device=device, dtype = torch.float), cl_p.reshape(3))

def calcularLossAtributo(nodo, radio):

    if nodo is None:
        return 0
    
    else:
        
        radio = radio.reshape(3)
        l2 = nn.MSELoss()
        
        return l2(nodo.radius, radio)

def decode_structure_fold(v, root, weight):
    def decode_node(v, count_level, node, weight):
        cl = nodeClassifier(v)
        _, label = torch.max(cl, 1)
        label = label.data
        #print("label", label)
        if label == 0 and createNode.count <= 22: ##output del classifier
            count_level.append("1")
            #if node.childs() != 0:
            lossEstructura = calcularLossEstructura(cl, node)
            radio = featuredec(v)

            lossAtrs = calcularLossAtributo( node, radio )
            return createNode(1,radio, cl_prob = weight * (lossEstructura + lossAtrs))
        elif label == 1 and createNode.count <= 22:
            right, radius = internaldec(v)
            #if node.childs() != 1:
            lossEstructura = calcularLossEstructura(cl, node)

            lossAtrs = calcularLossAtributo( node, radius )

            d = createNode(1, radius, cl_prob = weight * (lossEstructura + lossAtrs ) )
            count_level.append("1")
            
            if not node is None:
                if not node.right is None:
                    nodoSiguiente = node.right
                else:
                    nodoSiguiente = None
            else:
                nodoSiguiente = None
            d.right = decode_node(right, count_level, nodoSiguiente, weight * 0.7 )
            
            return d
        elif label == 2 and createNode.count <= 22:
            left, right, radius = bifdec(v)
            #if node.childs() != 2:
            lossEstructura = calcularLossEstructura(cl, node)

            lossAtrs = calcularLossAtributo( node, radius )

            d = createNode(1, radius, cl_prob = weight * (lossEstructura + lossAtrs ))
            count_level.append("1")
            

            if not node is None:
                if not node.right is None:
                    nodoSiguienteRight = node.right
                else:
                    nodoSiguienteRight = None

                if not node.left is None:
                    nodoSiguienteLeft = node.left
                else:
                    nodoSiguienteLeft = None
            else:
                nodoSiguienteRight = None
                nodoSiguienteLeft = None

            d.right = decode_node(right, count_level, nodoSiguienteRight, weight * 0.7)
            d.left = decode_node(left, count_level, nodoSiguienteLeft, weight * 0.7)
           
            return d
        
        
    count_level = []
    createNode.count = 0
    dec = decode_node(v, count_level, root, weight)
    return dec
        

t_list = ['test5.dat', 'test4.dat', 'test3.dat']
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


def main():

    epochs = 1000
    learning_rate = 1e-3

    leaf_encoder_opt = torch.optim.Adam(leafenc.parameters(), lr=learning_rate)
    non_leaf_encoder_opt = torch.optim.Adam(nonleafenc.parameters(), lr=learning_rate)
    class_opt = torch.optim.Adam(nodeClassifier.parameters(), lr=learning_rate)

    feature_decoder_opt = torch.optim.Adam(featuredec.parameters(), lr=learning_rate)
    bifurcation_decoder_opt = torch.optim.Adam(bifdec.parameters(), lr=learning_rate)
    internal_decoder_opt = torch.optim.Adam(internaldec.parameters(), lr=learning_rate)

    train_loss_avg = []
    ce_avg = []
    mse_avg = []
    l1_avg = []

    for epoch in range(epochs):
        train_loss_avg.append(0)
        ce_avg.append(0)
        mse_avg.append(0)
        l1_avg.append(0)
        weight = 1
        for data in data_loader:
            
            d_data = deserialize(data[0])
            enc_fold_nodes = encode_structure_fold(d_data).to(device)
            
            #print("encoded", enc_fold_nodes)
            
            decoded = decode_structure_fold(enc_fold_nodes, d_data, weight)
            #weight *= 1,1
            l = []
            loss_list = decoded.traverseInorderLoss(decoded, l)
            #breakpoint()
            total_loss = sum(loss_list)
            '''
            #print("decoded",decoded)
            ce_loss = nn.CrossEntropyLoss()
            mse_loss = nn.MSELoss()
            
            count = []
            in_n_nodes = len(d_data.count_nodes(d_data, count))
            #print("input n nodes: ", in_n_nodes)
            count = []
            out_n_nodes = len(decoded.count_nodes(decoded, count))
            
            #print("output n nodes: ", out_n_nodes)

            data_copy = d_data.cloneBinaryTree(d_data) #en las copias quedan sin modificar
            decoded_copy = decoded.cloneBinaryTree(decoded)

            in_radius_array = d_data.LevelOrder(d_data, 2)
            out_radius_array = decoded.LevelOrder(decoded, 2)
           
            radius_array = [(a, b) for a, b in zip(in_radius_array, out_radius_array) if a '#']
            radius_array = list(zip(*radius_array))
            try:
                mse = mse_loss(torch.cat(radius_array[1]), torch.tensor(radius_array[0]).to(device) ) 
            except:
                breakpoint()
            
            data_copy2 = data_copy.cloneBinaryTree(data_copy) #en las copias quedan sin modificar
            decoded_copy2 = decoded_copy.cloneBinaryTree(decoded_copy)
            list_original = data_copy.LevelOrder(data_copy, 0) #armo lista con la cantidad de hijos de cada nodo para el arbol original
            list_original = [ 3 if item == '#' else item for item in list_original]
            
            list_decoded = decoded.LevelOrder(decoded, 1) #armo lista con las probabilidades que da el clasificador para el arbol decodeado
            
            childs_array = [(a, b) for a, b in zip(list_original, list_decoded) if  b != '#']
            childs_array = list(zip(*childs_array))
            ce = ce_loss(torch.cat(childs_array[1]), torch.tensor(childs_array[0]).long().to(device) ) 
            
            multiplicador = (in_n_nodes/len(childs_array[0]))-1
            total_loss = ce  + 0.001*mse

            '''
            
            # Do parameter optimization
            leaf_encoder_opt.zero_grad()
            non_leaf_encoder_opt.zero_grad()
            feature_decoder_opt.zero_grad()
            bifurcation_decoder_opt.zero_grad()
            internal_decoder_opt.zero_grad()
            class_opt.zero_grad()

            total_loss.backward()

            leaf_encoder_opt.step()
            non_leaf_encoder_opt.step()
            feature_decoder_opt.step()
            bifurcation_decoder_opt.step()
            class_opt.step()
            internal_decoder_opt.step()

            train_loss_avg[-1] += total_loss
            #ce_avg [-1] += ce
            #mse_avg [-1] +=mse
            #l1_avg [-1] +=multiplicador

        if epoch % 1 == 0:
            print('Epoch [%d / %d] average reconstruction error: %f  ' % (epoch+1, epochs, train_loss_avg[-1]))

    #print(decoded_copy2.height(decoded_copy2))
    #decoded_copy2.traverseInorder(decoded_copy2)
    #copy = decoded_copy2.cloneWithoutZero(decoded_copy2) ## para cuando quedan nodos vacios en el arbol decodeado, no deberia pasar si esta bien entrenado
    #print(out_n_nodes)
    
    input = deserialize(iter(data_loader).next()[0])
    print(input)
    input.traverseInorder(input)
    encoded = encode_structure_fold(input).to(device)
    print("encoded", enc_fold_nodes)
    #decoded = decode_structure_fold(encoded)
    #decoded.traverseInorder(decoded)

    breakpoint()

if __name__ == "__main__":
    main()