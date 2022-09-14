from email import message
from logging import raiseExceptions
from tokenize import Double
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from vec3 import Vec3


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
    def __init__(self, value, radius, left = None, right = None, position = None, cl_prob= None, ce = None, mse = None):
        self.left = left
        self.data = value
        self.radius = radius
        self.position = position
        self.right = right
        self.prob = cl_prob
        self.mse = mse
        self.ce = ce
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

    def traverseInorderMSE(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderMSE(root.left, loss)
            loss.append(root.mse)
            self.traverseInorderMSE(root.right, loss)
            return loss

    def traverseInorderCE(self, root, loss):
        """
        traverse function will print all the node in the tree.
        """
        if root is not None:
            self.traverseInorderCE(root.left, loss)
            loss.append(root.ce)
            self.traverseInorderCE(root.right, loss)
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
        vertice = node[0][0][:3]
        rad = node[0][0][-1]
        num = node[1]
        
        #vertices.append((num, {'posicion': Vec3( vertice[0], vertice[1], vertice[2]), 'radio': rad} ))
        vertices.append((len(verticesCrudos),{'posicion': Vec3( vertice[0], vertice[1], vertice[2]), 'radio': rad}))
        verticesCrudos.append(vertice)


    G = nx.Graph()
    G.add_nodes_from( vertices )
    G.add_edges_from( lineas )
    
    a = nx.get_node_attributes(G, 'posicion')
   
    #for key in a.keys():
    #    a[key] = a[key].toNumpy()[0:2]

    #plt.figure(figsize=(20,10))
    #nx.draw(G, pos = a, node_size = 150, with_labels = True)
    #plt.show()
    return G

@count_fn
def createNode(data, radius, position = None, left = None, right = None, cl_prob = None, ce = None, mse=None):
        """
        Utility function to create a node.
        """
        return Node(data, radius, position, left, right, cl_prob, ce, mse)
 
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
        rad = radius.split(",")
        rad [0] = rad[0].replace('[','')
        rad [3] = rad[3].replace(']','')
        r = []
        for value in rad:
            r.append(float(value))
        #r =[float(num) for num in radius if num.isdigit()]
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
        self.l1 = nn.Linear(4, 16)
        self.l2 = nn.Linear(16, 32)
        self.tanh = nn.Tanh()

    def forward(self, input):
        rad = torch.tensor(input.radius)
        rad = torch.reshape(rad, (1,4)).to(device)
        radius = self.l1(rad)
        radius = self.tanh(radius)
        radius = self.l2(radius)
        radius = self.tanh(radius)
        
        return radius

class NonLeafEncoder(nn.Module):
    
    def __init__(self):
        super(NonLeafEncoder, self).__init__()
        self.l1 = nn.Linear(4,16)
        self.l2 = nn.Linear(16,32)

        self.left = nn.Linear(32, 32)
        self.right = nn.Linear(32, 32)
        
        self.encoder = nn.Linear(64, 32)
        self.tanh = nn.Tanh()


    def forward(self, input, left_input, right_input):
        
        radius = self.l1(torch.tensor(input.radius).reshape(1,4).to(device))
        radius = self.tanh(radius)
        radius = self.l2(radius)
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
        self.mlp = nn.Linear(32,32)
        self.lp2 = nn.Linear(32,32)
        self.mlp_right = nn.Linear(32,32)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(32,4)

    def forward(self, parent_feature):
        
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.lp2(vector)
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
        self.lp2 = nn.Linear(32,32)
        self.mlp_left = nn.Linear(32,32)
        self.mlp_right = nn.Linear(32,32)
        self.mlp2 = nn.Linear(32,4)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.lp2(vector)
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
        self.mlp2 = nn.Linear(16,8)
        self.tanh = nn.Tanh()
        self.mlp3 = nn.Linear(8,4)

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        vector = self.mlp2(vector)
        vector = self.tanh(vector)
        vector = self.mlp3(vector)
       

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
    #breakpoint()
    if original is None:
        return 0
    else:
        if original.childs() == 0:
            vector = [1, 0, 0]
        if original.childs() == 1:
            vector = [0, 1, 0]
        if original.childs() == 2:
            vector = [0, 0, 1]
    
    ce = nn.CrossEntropyLoss()
    
    return ce(cl_p, torch.tensor(vector, device=device, dtype = torch.float).reshape(1, 3))

def calcularLossAtributo(nodo, radio):

    if nodo is None:
        #return 0 #es cuando creo un nodo que esta "mal ubicado"

        return torch.tensor(1., device=device, requires_grad = True)
        ##return 1 y despues calculo otra loss con la suma de estos valores
    
    else:
        
        radio = radio.reshape(4)
        l2 = nn.MSELoss(reduction = 'sum')
       
        return l2(nodo.radius, radio )

def decode_structure_fold(v, root, weight):
    def decode_node(v, node, weight):
        cl = nodeClassifier(v)
        _, label = torch.max(cl, 1)
        label = label.data
        #print("label", label)
        if label == 0 and createNode.count <= 70: ##output del classifier
            #if node.childs() != 0:
            lossEstructura = calcularLossEstructura(cl, node)
            radio = featuredec(v)

            lossAtrs = calcularLossAtributo( node, radio )
            #return createNode(1,radio, cl_prob = weight * (lossEstructura + lossAtrs))
            return createNode(1,radio, ce = weight*lossEstructura,  mse = lossAtrs)
        elif label == 1 and createNode.count <= 70:
            right, radius = internaldec(v)
            #if node.childs() != 1:
            lossEstructura = calcularLossEstructura(cl, node)
            lossAtrs = calcularLossAtributo( node, radius )
            #breakpoint()
            #d = createNode(1, radius, cl_prob = weight * (lossEstructura + lossAtrs ) )
            if node is None:
                d = createNode(1, radius, cl_prob = lossAtrs , ce = weight*lossEstructura, mse= torch.tensor(0, device=device)) 
            else:
                d = createNode(1, radius, ce = weight*lossEstructura, mse = lossAtrs ) 
            
             
            if not node is None:
                if not node.right is None:
                    nodoSiguiente = node.right
                else:
                    nodoSiguiente = None
            else:
                nodoSiguiente = None
            d.right = decode_node(right, nodoSiguiente, 0.9*weight )
            
            return d
        elif label == 2 and createNode.count <= 70:
            left, right, radius = bifdec(v)
            #if node.childs() != 2:
            lossEstructura = calcularLossEstructura(cl, node)

            lossAtrs = calcularLossAtributo( node, radius )

            #d = createNode(1, radius, cl_prob = weight * (lossEstructura + lossAtrs ))
            if node is None:
                d = createNode(1, radius, cl_prob = lossAtrs, ce = weight*lossEstructura, mse = torch.tensor(0, device=device))
            else:
                d = createNode(1, radius, ce = weight*lossEstructura, mse = lossAtrs )
            
            if not node is None: #el nodo existe, me fijo si tiene hijo der/izq
                if not node.right is None:
                    nodoSiguienteRight = node.right
                else:
                    nodoSiguienteRight = None

                if not node.left is None:
                    nodoSiguienteLeft = node.left
                else:
                    nodoSiguienteLeft = None
            else: #el nodo no existe
                nodoSiguienteRight = None
                nodoSiguienteLeft = None

            d.right = decode_node(right, nodoSiguienteRight, 0.9*weight)
            d.left = decode_node(left, nodoSiguienteLeft, 0.9*weight )
           
            return d
        
    
    createNode.count = 0
    dec = decode_node(v, root, weight)
    return dec

def numerar_nodos(root, count):
    if root is not None:
        numerar_nodos(root.left, count)
        root.data = len(count)
        count.append(1)
        numerar_nodos(root.right, count)
        return 


def traversefeatures(root, features):
       
    if root is not None:
        traversefeatures(root.left, features)
        features.append(root.radius)
        traversefeatures(root.right, features)
        return features

def norm(root, minx, miny, minz, minr, maxx, maxy, maxz, maxr):
    #breakpoint()
    
    if root is not None:
        mx = minx.clone().detach()
        my = miny.clone().detach()
        mz = minz.clone().detach()
        mr = minr.clone().detach()
        Mx = maxx.clone().detach()
        My = maxy.clone().detach()
        Mz = maxz.clone().detach()
        Mr = maxr.clone().detach()

        root.radius[0] = (root.radius[0] - minx)/(maxx - minx)
        root.radius[1] = (root.radius[1] - miny)/(maxy - miny)
        root.radius[2] = (root.radius[2] - minz)/(maxz - minz)
        root.radius[3] = (root.radius[3] - minr)/(maxr - minr)
        
        norm(root.left, mx, my, mz, mr, Mx, My, Mz, Mr)
        norm(root.right, mx, my, mz, mr, Mx, My, Mz, Mr)
        return 

def normalize_features(root):
    features = []
    features = traversefeatures(root, features)
    
    x = [tensor[0] for tensor in features]
    y = [tensor[1] for tensor in features]
    z = [tensor[2] for tensor in features]
    r = [tensor[3] for tensor in features]
 
    norm(root, min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r))

    return 
        
t_list = ['ArteryObjAN1-7.dat', 'ArteryObjAN1-0.dat']
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

    epochs = 100
    learning_rate = 1e-3

    leaf_encoder_opt = torch.optim.Adam(leafenc.parameters(), lr=learning_rate)
    non_leaf_encoder_opt = torch.optim.Adam(nonleafenc.parameters(), lr=learning_rate)
    class_opt = torch.optim.Adam(nodeClassifier.parameters(), lr=learning_rate)

    feature_decoder_opt = torch.optim.Adam(featuredec.parameters(), lr=learning_rate)
    bifurcation_decoder_opt = torch.optim.Adam(bifdec.parameters(), lr=learning_rate)
    internal_decoder_opt = torch.optim.Adam(internaldec.parameters(), lr=learning_rate)

    #leaf_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=leaf_encoder_opt, gamma=0.9995)
    #non_leaf_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=non_leaf_encoder_opt, gamma=0.9995)
    #class_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=class_opt, gamma=0.9995)
    #feature_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=feature_decoder_opt, gamma=0.9995)
    #bifurcation_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=bifurcation_decoder_opt, gamma=0.9995)
    #internal_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=internal_decoder_opt, gamma=0.9995)


    train_loss_avg = []
    ce_avg = []
    mse_avg = []
    l1_avg = []
    lr_list = []
    l3_list = []
    for epoch in range(epochs):
        train_loss_avg.append(0)
        ce_avg.append(0)
        mse_avg.append(0)
        l1_avg.append(0)
        lr_list.append(0)
        l3_list.append(0)
        weight = 1
        for data in data_loader:
            
            d_data = deserialize(data[0])
            normalize_features(d_data)

            enc_fold_nodes = encode_structure_fold(d_data).to(device)
            decoded = decode_structure_fold(enc_fold_nodes, d_data, weight)
           
            l = []
            mse_loss_list = decoded.traverseInorderMSE(decoded, l)
            l = []
            ce_loss_list = decoded.traverseInorderCE(decoded, l)
            l = []
            loss_list = decoded.traverseInorderLoss(decoded, l)
            #print("mse", mse_loss_list)
            #print("ce", ce_loss_list)
            
            #for i in loss_list:
            #    if i == 1:
            #        breakpoint()
            '''
            l3 = torch.tensor(0, device = device)
            for element in loss_list:
                if element is not None:
                    #breakpoint()
                    torch.add (l3, element)
                    '''
            mse_loss = sum(mse_loss_list)
            #breakpoint()
            ce_loss = sum(ce_loss_list)
            loss_l3 = [a for a in loss_list if a is not None]
            l3 = sum(loss_l3)
            total_loss = mse_loss + ce_loss + l3
            #print("l3",l3)
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

            #leaf_scheduler.step()
            #non_leaf_scheduler.step()
            #class_scheduler.step()
            #feature_scheduler.step()
            #bifurcation_scheduler.step()
            #internal_scheduler.step()

            train_loss_avg[-1] += total_loss.item()
            ce_avg [-1] += ce_loss.item()
            mse_avg [-1] +=mse_loss.item()
            #lr_list [-1] += leaf_scheduler.get_last_lr()[-1]
            l3_list [-1] += l3

        if epoch % 1 == 0:
            print('Epoch [%d / %d] average reconstruction error: %f mse: %f, ce: %f, l3: %f' % (epoch+1, epochs, train_loss_avg[-1], mse_avg[-1], ce_avg[-1], l3_list[-1]))

    #print(decoded_copy2.height(decoded_copy2))
    #decoded_copy2.traverseInorder(decoded_copy2)
    #copy = decoded_copy2.cloneWithoutZero(decoded_copy2) ## para cuando quedan nodos vacios en el arbol decodeado, no deberia pasar si esta bien entrenado
    #print(out_n_nodes)
    
    input = deserialize(iter(data_loader).next()[0])
    normalize_features(input)
    input.traverseInorder(input)
    encoded = encode_structure_fold(input).to(device)
    print("encoded", enc_fold_nodes)
    decoded = decode_structure_fold(encoded, d_data, 1)
    count = []
    numerar_nodos(decoded, count)
    decoded.traverseInorder(decoded)
   
    G = arbolAGrafo (decoded)
    plt.figure()
    nx.draw(G, node_size = 150, with_labels = True)
    plt.show()
    fig = plt.plot(train_loss_avg) 
    #plt.savefig("loss.png")
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(mse_avg, label="MSE")
    ax2 = ax.twinx()
    ax2.plot(ce_avg, color="red", label="Cross Entropy")
    ax.legend(loc=1)
    ax.set_ylim(0, max(mse_avg))

    ax2.legend(loc=3)
    ax2.set_ylim(0, max(ce_avg))
    #plt.savefig("mse-ce.png")
    plt.show()
    breakpoint()
   

if __name__ == "__main__":
    main()