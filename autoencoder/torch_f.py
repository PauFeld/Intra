from __future__ import absolute_import
import collections
import torch

class Fold(object):

    class Node(object):
        def __init__(self, op, step, index, *args):
            self.op = op
            self.step = step
            self.index = index
            self.args = args
            self.split_idx = -1
            self.batch = True

        
        def split(self, num):
            u"""Split resulting node, if function returns multiple values."""
            nodes = []
            for idx in range(num):
                nodes.append(Fold.Node(
                    self.op, self.step, self.index, *self.args))
                nodes[-1].split_idx = idx
            return tuple(nodes)
            

        def nobatch(self):
            self.batch = False
            return self

        def get(self, values):
            if self.split_idx >= 0:
                return values[self.step][self.op][self.split_idx][self.index]
            else:
                return values[self.step][self.op][self.index]

        def __repr__(self):
            return u"[%d:%d]%s" % (
                self.step, self.index, self.op)

    def __init__(self, volatile=False, cuda=False, variable=True):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        self.total_nodes = 0
        self.volatile = volatile
        self._cuda = cuda
        self._variable = variable

    def __repr__(self):
        return str(self.steps.keys())

    def cuda(self):
        self._cuda = True
        return self

    def add(self, op, *args):
        u"""Add op to the fold."""

        self.total_nodes += 1
        # si el nodo no fue visitado antes 
        if args not in self.cached_nodes[op]:
            step = max([0] + [arg.step + 1 for arg in args if isinstance(arg, Fold.Node)])
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
        return self.cached_nodes[op][args]


    def _batch_args(self, arg_lists, values, op):
        res = []
        for arg in arg_lists:
            r = []
            #si es un nodo de fold
            if isinstance(arg[0], Fold.Node):
                if arg[0].batch:
                    for x in arg:
                        r.append(x.get(values))
                    res.append(torch.stack(r))
                
                #nunca uso este caso
                '''
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError(u"Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    res.append(x.get(values))
                    '''
            else:           
                #si es un tensor de atributos     
                if isinstance(arg[0], torch.Tensor):  
                    var = torch.stack(arg)
                    res.append(var)
                
                #si es un nodo de arbol
                else:
                    if op != "classifyLossEstimator" and op != "calcularLossAtributo": #en caso de que op sea alguna red
                        var = arg[0].radius
                    elif op == "calcularLossAtributo": #en caso de estar calculano mse
                        var = [a.radius for a in arg]
                    else:
                        var = [a.childs() for a in arg] #en caso de estar calculando cross entropy
                    res.append(var)
                  
        return res

    def apply(self, nn, nodes):
        u"""Apply current fold to given neural module."""
        values = {}
        for step in sorted(self.steps.keys()):
            
            values[step] = {}
            for op in self.steps[step]:
                func = getattr(nn, op)
                ##junto los atributos de los nodos que estan en el mismo step y op
                try:                    
                    batched_args = self._batch_args(
                        zip(*self.steps[step][op]), values, op)
                except Exception:
                    print("Error while executing node %s[%d] with args: %s" % (op, step, self.steps[step][op]))
                    raise

                res = func(*batched_args)
                
                if isinstance(res, (tuple, list)):
                    values[step][op] = []
                    for x in res:
                        #values[step][op].append(torch.chunk(x, arg_size))
                        values[step][op].append(x)
                else:
                    if len(res.shape) == 1 and op != 'vectorAdder':
                        values[step][op] = res.reshape(-1, 4)
                    else: #los vectores de output del clasificador tienen tres elementos, no hago el reshape
                        values[step][op] = res
                       
        try:
            return self._batch_args(nodes, values, op)
        except Exception:
            print("cannot batch")
            raise