import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    # build graph edges
    nodes, edges = trace(root)
    for n in nodes:
        # unique id based on python object address
        uid = str(id(n))
        # for any value in the graph, create a rectangular 'record' node
        dot.node(name=uid, label="{%s |data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of an operation, create a node for it
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        # calculate the gradient of self
        self._backward = lambda: None
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+', )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        tan_h = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(tan_h,(self,),'tanh')

        def _backward():
            self.grad += (1 - tan_h**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
            
class MLP:
    """
    
    nouts=[8, 8, 1] → 3 层:8->8->1,列表长度 = 网络深度，列表元素 = 每层神经元个数。
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    # nural netwrok
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(8, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh()
    o.label = 'o'


    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(8, label='b')

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b; n.label = 'n'
    e = (2 * n).exp()
    o =(e-1)/(e+1)
    o.label = 'o'
    
    o.backward()
    dot = draw_dot(o)
    dot.render('graph', format='svg')

    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
    b = torch.Tensor([8.0]).double(); b.requires_grad = True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    print(o.data.item())
    o.backward()

    print('---')
    print('x2', x2.grad.item())
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())

    print('---')
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    o = n(x)
    o.backward()
    print(n.parameters())
    # print(n(x))

    draw_dot(o).render('graph', format='svg')

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = [n(x) for x in xs]
    print(ypred)
    # all the loss
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    print('loss', loss)
    loss.backward()
    print(n.layers[0].neurons[0].w[0].grad)
    print(n.layers[0].neurons[0].w[0].data)
    for p in n.parameters():
        p.data += -0.01 * p.grad
    print(n.layers[0].neurons[0].w[0].data)

    ypred = [n(x) for x in xs]
    print('ypred',ypred)
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    print('loss', loss)

    for k in range(100):
        ypred = [n(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()
        for p in n.parameters():
            p.data += -0.1 * p.grad
        print(k, loss)
    print('---')
    print(ypred)
    draw_dot(loss).render('graph', format='svg')

