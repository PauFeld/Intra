import collections

import torch
#from torch.autograd import Variable


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
            """Split resulting node, if function returns multiple values."""
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
            print("v", values[self.step][self.op])
            return values[self.step][self.op].get(self.index, self.split_idx)

        def __repr__(self):
            return "[%d:%d]%s" % (
                self.step, self.index, self.op)

    class ComputedResult(object):
        def __init__(self, batch_size, batched_result):
            self.batch_size = batch_size
            self.result = batched_result
            if isinstance(self.result, tuple):
                self.result = list(self.result)

        def try_get_batched(self, nodes):
            all_are_nodes = all(isinstance(n, Fold.Node) for n in nodes)
            num_nodes_is_equal = len(nodes) == self.batch_size

            if not all_are_nodes or not num_nodes_is_equal:
                return None

            valid_node_sequence = all(
                nodes[i].index < nodes[i + 1].index  # Indices are ordered
                and nodes[i].split_idx == nodes[i + 1].split_idx  # Same split index
                and nodes[i].step == nodes[i + 1].step  # Same step
                and nodes[i].op == nodes[i + 1].op  # Same op
                for i in range(len(nodes) - 1))
            if not valid_node_sequence:
                return None

            if nodes[0].split_idx == -1 and not isinstance(self.result, tuple):
                return self.result
            elif nodes[0].split_idx >= 0 and not isinstance(self.result[nodes[0].split_idx], tuple):
                return self.result[nodes[0].split_idx]
            else:
                # This result was already chunked.
                return None

        def get(self, index, split_idx=-1):
            if split_idx == -1:
                if not isinstance(self.result, tuple):
                    self.result = torch.chunk(self.result, self.batch_size)
                return self.result[index]
            else:
                if not isinstance(self.result[split_idx], tuple):
                    self.result[split_idx] = torch.chunk(self.result[split_idx], self.batch_size)
                return self.result[split_idx][index]

    def __init__(self, cuda=False):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        self.total_nodes = 0
        self._cuda = cuda

    def cuda(self):
        self._cuda = True
        return self

    
    def add(self, op, *args):
       
        """Add op to the fold."""
        #print("args", args)
        self.total_nodes += 1
        
        #for arg in args:
            #print("arg", arg) #if isinstance(arg, Fold.Node)
            #print("op", op)
        #args = [arg for arg in args if arg is not None]
        #print("args", args)
        if not all([isinstance(arg, (
                Fold.Node, int, torch.Tensor, None)) for arg in args]):
            raise ValueError(
                "All args should be Tensor, int or Node, got: %s" % str(args))
        
        
            
        #cuando el nodo izq o der no exisate se pasa un none y eso es lo que crea el problema
        if args not in self.cached_nodes[op]:
            step = max([0] + [arg.step + 1 for arg in args
                              if isinstance(arg, Fold.Node)])
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            #print("node", node)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node
            #print("cached", self.cached_nodes)
        #print("return", self.cached_nodes[op][args])
        return self.cached_nodes[op][args]
   
    

    '''
    def _batch_args(self, arg_lists, values):
        res = []
        for arg in arg_lists:
            r = []
            print("arg", arg)
            
            if all(isinstance(arg_item, Fold.Node) for arg_item in arg):
                assert all(arg[0].batch == arg_item.batch
                           for arg_item in arg[1:])

                if arg[0].batch:
                    batched_arg = values[arg[0].step][arg[0].op].try_get_batched(arg)
                    if batched_arg is not None:
                        print("batched", batched_arg)
                        res.append(batched_arg)
                    else:
                        print("a", arg)
                        for arg_item in arg:
                            print("arg_item", arg_item.get(values))
                            
                        res.append(torch.cat([arg_item.get(values) for arg_item in arg], 0))
                        print("cat", [arg_item.get(values)
                                       for arg_item in arg])
                    
                else:
                   
                    for arg_item in arg[1:]:
                        if arg_item != arg[0]:
                            raise ValueError("Can not use more then one of nobatch argument, got: %s." % str(arg_item))
                    res.append(arg[0].get(values))
                print("Res", res)
            elif all(isinstance(arg_item, int) for arg_item in arg):
                if self._cuda:
                    var =  torch.Tensor(arg, device = self._cuda)
                else:
                    var =  torch.Tensor(arg)
                res.append(var)
                
            else:
                for arg_item in arg:
                    print("arg", arg_item)
                    print("res", res)
                    if isinstance(arg_item, Fold.Node):
                        assert arg_item.batch
                        r.append(arg_item.get(values))
                    elif isinstance(arg_item, (torch.Tensor)):
                        r.append(arg_item)
                    else:
                        raise ValueError(
                            'Not allowed to mix Fold.Node/Tensor with int')
                res.append(torch.cat(r, 0))
        return res
    '''
    def _batch_args(self, arg_lists, values):
        res = []
        for arg in arg_lists:
            r = []
            if isinstance(arg[0], Fold.Node):
                #print("if", arg)
                if arg[0].batch:
                    #print("arg0", arg[0])
                    for x in arg:
                        r.append(x.get(values))
                        #print("r",r)
                        #print("x",x.get(values))
                    res.append(torch.cat(r, 0))
                else:
                    for i in range(2, len(arg)):
                        if arg[i] != arg[0]:
                            raise ValueError("Can not use more then one of nobatch argument, got: %s." % str(arg))
                    x = arg[0]
                    print("x", x)
                    res.append(x.get(values))
                    
            else:
                #print("else", arg)
                # Below is what this extension changes against the original version:
                #   We make Fold handle float tensor
                try:
                    if (isinstance(arg[0], torch.Tensor)):
                        var = torch.cat(arg, 0)
                    else:
                        var = torch.cat(torch.tensor(arg, 0))
                    if self._cuda:
                        var = var.cuda()
                    res.append(var)
                    #print("var", var)
                except:
                    print("Constructing float tensor from %s" % str(arg))
                    raise
        print("res", res)
        return res
   

    def apply(self, nn, nodes): #nn es la red y nodes el fold
        """Apply current fold to given neural module."""
        values = {}
        print("keys", self.steps[0])
        for step in sorted(self.steps.keys()):
            print("step", step)
            values[step] = {}
            for op in self.steps[step]:
                print("op", op)
                func = getattr(nn, op)
                try:
                    batched_args = self._batch_args(zip(*self.steps[step][op]), values)
                    print("batched_args", batched_args)
                    print("batched_args", batched_args[0])
                    
                except Exception:
                    print("Error while executing node %s[%d] with args: %s" % (
                        op, step, self.steps[step][op][0]))
                    raise
                if batched_args:
                    arg_size = batched_args[0].size()[0]
                    print("arg_size", arg_size)
                else:
                    arg_size = 1
                res = func(*batched_args)
                values[step][op] = Fold.ComputedResult(arg_size, res)
        try:
            return self._batch_args(nodes, values)
        except Exception:
            print("Retrieving %s" % nodes)
            for lst in nodes:
                if isinstance(lst[0], Fold.Node):
                    print(', '.join([str(x.get(values).size()) for x in lst]))
            raise

    def __str__(self):
        result = ''
        for step in sorted(self.steps.keys()):
            result += '%d step:\n' % step
            for op in self.steps[step]:
                first_el = ''
                for arg in self.steps[step][op][0]:
                    if first_el: first_el += ', '
                    if isinstance(arg, (torch.Tensor)):
                        first_el += str(arg.size())
                    else:
                        first_el += str(arg)
                result += '\t%s = %d x (%s)\n' % (
                    op, len(self.steps[step][op]), first_el)
        return result

    def __repr__(self):
        return str(self)


class Unfold(object):
    """Replacement of Fold for debugging, where it does computation right away."""

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
            raise ValueError("Expected that nn argument passed to constructor and passed to apply would match.")
        result = []
        for n in nodes:
            result.append(torch.cat([self._arg(a) for a in n]))
        return result