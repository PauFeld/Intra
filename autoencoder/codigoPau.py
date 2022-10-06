class Node:
    ...
    def toGraph( self, graph, index, proc=True ):
        
        radius = self.radius.numpy()
        graph.add_nodes_from( [ (index, {'posicion': self.radius[0:3], 'radio': self.radius[3] } ) ])

        if self.right is not None:
            leftIndex = self.right.toGraph( graph, index + 1)

            graph.add_edge( index, index + 1 )
            if proc:
                nx.set_edge_attributes( graph, {(index, index+1) : {'procesada':False}})
        
            if self.left is not None:
                retIndex = self.left.toGraph( graph, leftIndex )

                graph.add_edge( index, leftIndex)
                if proc:
                    nx.set_edge_attributes( graph, {(index, leftIndex) : {'procesada':False}})
            
            else:
               return leftIndex

        else:
            return index + 1



def plotTree( root ):
    graph = nx.Graph()
    root.toGraph( graph, 0 )

    p = mp.plot( np.array([ graph.nodes[v]['posicion'] for v in graph.nodes]), shading={'point_size':2}, return_plot=True)

    for arista in graph.edges:
        p.add_lines( graph.nodes[arista[0]]['posicion'], graph.nodes[arista[1]]['posicion'])

# y para plotear haces esto:
plotTree(arbol)