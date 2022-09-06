

def subtree(node, relationships):
        return {
            v: subtree(v, relationships) 
            for v in [x[0] for x in relationships if x[1] == node]
        }


# (child, parent) pairs where -1 means no parent    
flat_tree = [
         (1, -1),
         #(2, 1),
         #(3, 2),
         (4,1),
         (5, 4),
         (6, 5),
         (10, 6),
         (11,10),
         (7, 6),
         (8, 4),
         (9, 8),
         
        ]

a =    subtree(-1, flat_tree)
print(a.values())
b = list(a.values())
print(b[0].values())


class Node():
    def __init__(self, data=None):
        self.data = data
        self.children = []

    def __repr__(self, indent=""):
        return (indent + repr(self.data) + "\n"
                + "".join(child.__repr__(indent+"  ") 
                          for child in self.children))

def create_tree(edges):
    # Get all the unique keys into a set
    node_keys = set(key for keys in edges for key in keys)
    # Create a Node instance for each of them, keyed by their key in a dict:
    nodes = { key: Node(key) for key in node_keys }
    # Populate the children attributes from the edges
    for parent, child in edges:
        nodes[parent].children.append(nodes[child])
        # Remove the child from the set, so we will be left over with the root
        node_keys.remove(child)
    # Get the root from the set, which at this point should only have one member
    for root_key in node_keys:  # Just need one
        return nodes[root_key]

# Example run
edges = [[1,4],[1,3],[1,2],[3,5],[3,6],[3,7]]
root = create_tree(edges)

print(root)
