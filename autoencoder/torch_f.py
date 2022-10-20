from __future__ import absolute_import
import collections

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#from itertools import izip


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
                #print("step", self.step)
                #print("index", self.index)
                #print("value", values[self.step][self.op])
                #print("value", values[self.step][self.op][self.index])
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

        #print(args)

        self.total_nodes += 1
        
        # si el nodo no fue visitado antes 
        if args not in self.cached_nodes[op]:
            # 
            step = max([0] + [arg.step + 1 for arg in args if isinstance(arg, Fold.Node)])
            #print("step", step)
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
        return self.cached_nodes[op][args]


    def _batch_args(self, arg_lists, values, op):
        res = []
        #print("v", values)
        for arg in arg_lists:
            #print('recorro args', arg)
            r = []
            if isinstance(arg[0], Fold.Node):
                if arg[0].batch:
                    for x in arg:
                        #print("x", x)
                        r.append(x.get(values))
                        #print("r", r)
                    
                    res.append(torch.stack(r))
                    #res.append(r)
                    #print("ld", res)
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError(u"Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    res.append(x.get(values))
                    #print("lg", res)
                

            else:
                #print("else")
                if isinstance(arg[0], torch.Tensor):  
                    try:
                        #print("antes de cat", arg)
                        var = torch.stack(arg)
                        #print("despues de cat", var)
                        res.append(var)
                        #print("lc", res)
                    except:
                        print("Constructing LongTensor from %s" % str(arg))
                        raise
                else:
                    #print("arg es nodo")
                    #print("radio", arg[0].radius)
                    #print("op", op)
                    if op != "classifyLossEstimator" :
                    #var = torch.cat(arg[0].radius,0)
                        var = arg[0].radius
                    else:
                        #print("arg nodo", arg)
                        var = [a.childs() for a in arg]
                        #print("var", var)
                    res.append(var)
                    #print("res", res)
        
        #print("RES", res)           
        return res

    def apply(self, nn, nodes):
        u"""Apply current fold to given neural module."""
        values = {}
        for step in sorted(self.steps.keys()):
            #print('itera',step)
            
            values[step] = {}
#            print("vs", values)
#            print("vs", values[step])
            for op in self.steps[step]:
                func = getattr(nn, op)
                #print("op", op)

                ##junto los atributos de los nodos
                ##si estoy en op classify  loss, necesito el nodo directamente y no los atributos
                try:                    
                    batched_args = self._batch_args(
                        zip(*self.steps[step][op]), values, op)
                    #print("bc", batched_args)
                except Exception:
                    print("Error while executing node %s[%d] with args: %s" % (op, step, self.steps[step][op]))
                    raise
                #print('batched_args',batched_args)
                if batched_args is not None:
                    #print("arg", batched_args)
                    #arg_size = batched_args[0].size()[0]
                    arg_size = len(batched_args[0])
                    #print(arg_size)
                else:
                    arg_size = 1
                #print("batched args", batched_args)
                #print("batched args", len(batched_args))
                
                #batched_args = batched_args[0].reshape(4,-1).T
                #if op != "classifyLossEstimator" :
                    #print("b 4", batched_args)
                    #batched_args = [b.reshape(4, -1).T for b in batched_args]
                
                #else:
                #    print("bat", batched_args[0])
                    #batched_args = [batched_args[0], nodes]
                
                #print("batched args reshaped", batched_args)
                res = func(*batched_args)
                #print("paso por la red", res)
#                print("is", isinstance(res, (tuple, list)))
                if isinstance(res, (tuple, list)):
                    #print('res if',res)
                    values[step][op] = []
                    for x in res:
                        #values[step][op].append(torch.chunk(x, arg_size))
                        values[step][op].append(x)
                    #print("values", values[step][op])
                else:
                    #print('else res',res)
                    #print('res shape',res.shape)
                    #print(arg_size)
                    #print("len", len(res.shape))
                    #print("len", len(res))
                    #print("op", op)
                    if len(res.shape) == 1 and op != 'vectorAdder':
                        #values[step][op] = torch.split(res, arg_size, 0)
                        values[step][op] = res.reshape(-1, 4)
                    else:
                        values[step][op] = res
                
                    #print("chunk", len(values[step][op]))
                    '''
                    for c in range(len(ch)):
                        values[step][op][c] = ch[c]
                    print('res',values[step][op])
                    '''
                    
   
                    
        try:
            return self._batch_args(nodes, values, op)
        except Exception:
            print("Retrieving %s" % nodes)
            for lst in nodes:
                if isinstance(lst, Fold.Node):
                    print(', '.join([str(x.get(values).size()) for x in lst]))
            raise


class Unfold(object):
    u"""Replacement of Fold for debugging, where it does computation right away."""

    class Node(object):

        def __init__(self, tensor):
            self.tensor = tensor

        def __repr__(self):
            return str(self.tensor)

        def nobatch(self):
            return self

        def split(self, num):
            return [Unfold.Node(self.tensor[i]) for i in range(num)]

    def __init__(self, nn, volatile=False, cuda=False):
        self.nn = nn
        self.volatile = volatile
        self._cuda = cuda

    def cuda(self):
        self._cuda = True
        return self

    def _arg(self, arg):
        if isinstance(arg, Unfold.Node):
            return arg.tensor
        elif isinstance(arg, int):
            if self._cuda:
                return torch.cuda.LongTensor([arg])
            else:
                return torch.LongTensor([arg])
        else:
            return arg

    def add(self, op, *args):
        values = []
        for arg in args:
            values.append(self._arg(arg))
        res = getattr(self.nn, op)(*values)
        return Unfold.Node(res)

    def apply(self, nn, nodes):
        if nn != self.nn:
            raise ValueError(u"Expected that nn argument passed to constructor and passed to apply would match.")
        result = []
        for n in nodes:
            result.append(torch.cat([self._arg(a) for a in n]))
        return result