import subprocess
import graphviz
from collections import defaultdict
import seaborn as sns
import matplotlib
import matplotlib.colors
from itertools import cycle
from numbers import Real as FloatingPointError

from typing import TypeVar
from collections.abc import Callable

from .vscode_theme import is_vscode_dark_theme

GraphType = TypeVar('Graph') 

# def random_color():
#     return '#'+''.join(random.sample('0123456789ABCDEF', 6))

def _get_color(n, lightness=0.4):
    color_cycle = cycle([matplotlib.colors.to_hex(c) for c in sns.husl_palette(n, l=lightness)])
    for color in color_cycle:
        yield color

def _format_rate(rate):
    if rate == round(rate):
        return f"{rate:.2f}"
    else:
        return f"{rate:.2e}"



def plot_graph(graph:GraphType, 
               subgraphfun:Callable=None, 
               by_state:Callable=None, 
               by_index:Callable=None, 
               max_nodes:int=100, 
               dark:bool=True,
               constraint:bool=True, ranksep:float=1, nodesep:float=1, rankdir:str="LR",
               size:tuple=(7, 7), fontsize:int=12, rainbow:bool=True, penwidth:FloatingPointError=1,
               seed:int=1,                
               **kwargs) -> graphviz.Digraph:
    """
    Plot a graph using graphviz.

    ----------
    graph : 
        _description_
    subgraphfun : 
        Callback function defining subgraph clusters. Must take a state as input and produce a string that serve as subgraph label. None by default.
    max_nodes : 
        Maximum number of vertices for graphs to plot, by default 100
    dark : 
        Whether to use dark mode for the graph, by default True
    rainbow : 
        Color edges randomly, by default True
    size : 
        Graphviz size, by default (7, 7)
    constraint : 
        Graphviz constaint, by default True
    ranksep : 
        Graphviz ranksep, by default 1
    nodesep : 
        Graphviz nodesep, by default 1
    rankdir : 
        Graphviz rankdir, by default "LR"
    fontsize : 
        Graphviz fontsize, by default 12
    penwidth : 
        Graphviz penwidth, by default 1

    Returns
    -------
    :
        Graphviz object for Jupyter notebooks display
    """

    # try: 
    #     subprocess.check_call('dot', timeout=0.1)#.output.startswith('There is no layout engine support for "dot"'):
    # except:
    subprocess.check_call(['dot', '-c']) # register layout engine

    # backwards comp
    if by_state is None and subgraphfun is not None:
        by_state = subgraphfun

    if by_state and by_index:
        assert "Do not use both by_index and by_state"

    if dark is None:
        dark = is_vscode_dark_theme()

    if dark:
        edge_color = '#e6e6e6'
        node_edgecolor = '#888888'
        node_fillcolor = "#c6c6c6"
        start_edgecolor = 'black'
        start_fillcolor = '#777777'
        abs_edgecolor = 'black'
        abs_fillcolor = '#777777'
        aux_edgecolor = 'black'
        aux_fillcolor = '#3e3e3e'
        bgcolor = '#1F1F1F'
        subgraph_label_fontcolor = '#e6e6e6'
        subgraph_bgcolor='#2e2e2e'
        subgraph_edgecolor='#e6e6e6'
        husl_colors = _get_color(10, lightness=0.7)
    else:
        edge_color = '#009900'
        node_edgecolor='black'
        node_fillcolor='#eeeeee'
        edge_color='black' 
        start_edgecolor='black'
        start_fillcolor='#bbbbbb'
        abs_edgecolor='black'
        abs_fillcolor='#bbbbbb'
        aux_edgecolor='black'
        aux_fillcolor='#bbbbbb'
        bgcolor='transparent'
        subgraph_label_fontcolor = 'black'
        subgraph_bgcolor='white'
        subgraph_edgecolor='black'
        husl_colors = _get_color(10, lightness=0.4)

    if graph.vertices_length() > max_nodes:
        print(f"Graph has too many nodes ({graph.vertices_length()}). Please set max_nodes to a higher value.")
        return None

    graph_attr = dict(compound='true', newrank='true', pad='0.5', 
                      ranksep=str(ranksep), nodesep=str(nodesep), 
                      bgcolor=bgcolor, rankdir=rankdir, ratio="auto",
                      size=f'{size[0]},{size[1]}',
                      start=str(seed),
                      fontname="Helvetica,Arial,sans-serif", **kwargs)
    node_attr = dict(style='filled', color='black',
                     fontname="Helvetica,Arial,sans-serif", 
                     fontsize=str(fontsize), 
                     fillcolor=str(node_fillcolor))
    edge_attr = dict(constraint='true' if constraint else 'false',
                     style='filled', labelfloat='false', labeldistance='0',
                     fontname="Helvetica,Arial,sans-serif", 
                     fontsize=str(fontsize), penwidth=str(penwidth))    
    dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        for edge in vertex.edges():
            if rainbow:
                color = next(husl_colors)
                # color = random_color()
            else:
                 color = edge_color
            dot.edge(str(vertex.index()), str(edge.to().index()), 
                   xlabel=_format_rate(edge.weight()), color=color, fontcolor=color)

    subgraph_attr = dict(rank='same',
                         style='filled', 
                         fillcolor=subgraph_bgcolor, 
                         color=subgraph_edgecolor,
                         fontcolor=subgraph_label_fontcolor)
    subgraphs = defaultdict(list)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        if i == 0:
            dot.node(str(vertex.index()), 'S', 
                     style='filled', edge_color=start_edgecolor, fillcolor=start_fillcolor)
        elif not vertex.state().sum() and vertex.rate() == 1 and len(vertex.edges()) == 1:
            dot.node(str(vertex.index()), 'AUX', 
                     style='filled', edge_color=aux_edgecolor, fillcolor=aux_fillcolor)
        elif not vertex.edges():
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())), 
                     style='filled', edge_color=abs_edgecolor, fillcolor=abs_fillcolor)
        else:
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())),
                     style='filled', edge_color=node_edgecolor, fillcolor=node_fillcolor)

        if i != 0:
            if by_state:
                subgraphs[f'cluster_{by_state(vertex.state())}'].append(i)
            elif by_index:
                subgraphs[f'cluster_{by_index(vertex.index())}'].append(i)

    if by_state or by_index:
        for sglabel in subgraphs:
            subgraph_attr['label'] = sglabel.replace('cluster_', '')
            with dot.subgraph(name=sglabel, graph_attr=subgraph_attr) as c:
                for i in subgraphs[sglabel]:
                    vertex = graph.vertex_at(i)
                    # c.node(str(vertex.index()), ','.join(map(str, vertex.state())))
                    c.node(str(vertex.index()))
    return dot
