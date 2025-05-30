import math, subprocess, io
import numpy as np
import matplotlib.pyplot as plt
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
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+', )
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')
    
    def __sub__(self, other):
        return Value(self.data - other.data, (self, other), '-')
    
    def __truediv__(self, other):
        return Value(self.data / other.data, (self, other), '/')
    
    def __pow__(self, other):
        return Value(self.data ** other.data, (self, other), '**')
    
    def tanh(self):
        x = self.data
        tan_h = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        return Value(tan_h,(self,),'tanh')

def f(x):
    return 3*x**2 - 4*x + 5

def lol():
    h = 0.0001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e';
    d = e + c; d.label = 'd';
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e';
    d = e + c; d.label = 'd';
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L
    print((L2 - L1)/Value(h))
    return L1
if __name__ == "__main__":
    # xs = np.arange(-5, 5, 0.25)
    # ys = f(xs)
    # plt.plot(xs, ys)
    # # 把画布保存到内存缓冲区
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # plt.close()             # 关闭 figure，节省内存
    # buf.seek(0)

    # # 通过 subprocess 把 PNG 数据发给 kitty icat
    # subprocess.run(
    #     ['kitty', '+kitten', 'icat'],
    #     input=buf.read()
    # )
    lol()

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e';
    d = e + c; d.label = 'd';
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    L.grad = 1
    d.grad = -2.0
    f.grad = 4.0
    c.grad = d.grad * 1.0
    e.grad = d.grad * 1.0
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    dot = draw_dot(L)
    dot.render('graph', format='svg')

    a.data += 0.01 * a.grad
    b.data += 0.01 * b.grad
    c.data += 0.01 * c.grad
    f.data += 0.01 * f.grad
    e = a * b;
    d = e + c;
    L = d * f
    print(L)

    
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
    o.grad = 1.0
    # do/dn = (1 - o^2)
    n.grad = o.grad * (1 - o.data**2)
    b.grad = n.grad * 1.0
    x1w1x2w2.grad = n.grad * 1.0
    x1w1.grad = x1w1x2w2.grad * 1.0
    x2w2.grad = x1w1x2w2.grad * 1.0
    x1.grad = x1w1.grad * w1.data
    x2.grad = x2w2.grad * w2.data
    w1.grad = x1w1.grad * x1.data
    w2.grad = x2w2.grad * x2.data
    
    dot = draw_dot(o)
    dot.render('graph', format='svg')