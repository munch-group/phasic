
import graphviz
import random
from collections import defaultdict
from IPython.display import display

def random_color():
    return '#'+''.join(random.sample('0123456789ABCDEF', 6))


def format_rate(rate):
    if rate == round(rate):
        return f"{rate:.2f}"
    else:
        return f"{rate:.2e}"


def plot_graph(graph, constraint=True,ranksep=1, 
               nodesep=1,
               subgraphs=False, 
                #  splines=False, 
            #    subgraphfun=lambda state, index: ','.join(map(str, state[:-1])), 
            size=(7, 7), 
               subgraphfun=None,
               fontsize=12, rankdir="LR", align=False, nodecolor='white', 
               edgecolor='black', 
            #    edgecolor='rainbow', 
               penwidth=1, **kwargs):

    graph_attr = dict(
        compound='true',
        newrank='true',
        pad='0.5',
        ranksep=str(ranksep),
        nodesep=str(nodesep),
        bgcolor='transparent',
        rankdir=rankdir,
        # splines='true' if splines else 'false',
        size=f'{size[0]},{size[1]}',
        fontname="Helvetica,Arial,sans-serif",
        ratio="auto",
        **kwargs
    )
    node_attr = dict(
        style='filled',
        color='black',
    	fontname="Helvetica,Arial,sans-serif", 
        fontsize=str(fontsize), 
        fillcolor=str(nodecolor),
    )
    edge_attr = dict(
        constraint='true' if constraint else 'false',
        style='filled',
        labelfloat='false', 
        labeldistance='0',
    	fontname="Helvetica,Arial,sans-serif", 
        fontsize=str(fontsize), 
        penwidth=str(penwidth),
    )
    dot = graphviz.Digraph(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    )
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        for edge in vertex.edges():
            if edgecolor == 'rainbow':
                color = random_color()
            else:
                 color = edgecolor
            dot.edge(str(vertex.index()), str(edge.to().index()), 
                   xlabel=format_rate(edge.weight()), color=color, fontcolor=color)

    subgraph_attr = dict(
        rank='same',
        style='filled',
        color='whitesmoke',
    )
    subg = defaultdict(list)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        if i == 0:
            dot.node(str(vertex.index()), 'S', 
                     style='filled', color='black', fillcolor='#eeeeee')
        elif not vertex.edges():
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())), 
                     style='filled', color='black',fillcolor='#eeeeee')
        elif subgraphfun is not None:
            subg[f'cluster_{subgraphfun(vertex.state())}'].append(i)
        else:
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())))
    if subgraphfun is not None:
        for sglabel in subg:
            subgraph_attr['label'] = sglabel.replace('cluster_', '')
            with dot.subgraph(name=sglabel, graph_attr=subgraph_attr) as c:
                for i in subg[sglabel]:
                    vertex = graph.vertex_at(i)
                    c.node(str(vertex.index()), ','.join(map(str, vertex.state())))
    return dot
