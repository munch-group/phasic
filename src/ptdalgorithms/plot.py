
import subprocess
import graphviz
import random
from collections import defaultdict
from IPython.display import display
import seaborn as sns
import matplotlib.colors
from itertools import cycle

from typing import Any, TypeVar, List, Tuple, Dict, Union
from collections.abc import Sequence, MutableSequence, Callable

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

_theme = 'dark'

def set_theme(theme:str):
    """
    Set the default theme for the graph plotter.
    The theme can be either 'dark' or 'light'. The default theme is 'dark'.

    Parameters
    ----------
    theme : 
        _description_
    """
    global _theme
    _theme = theme


GraphType = TypeVar('Graph') 


# def plot_graph (*args, **kwargs):
#     try:
#         _plot_graph(*args, **kwargs)
#     except Exception as e:
#         subprocess.check_call(['dot', '-c']) # register layout engine
#         _plot_graph(*args, **kwargs)


def plot_graph(graph:GraphType, 
               subgraphfun:Callable=None, max_nodes:int=100, 
               theme:str=None,
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
    theme : 
        Style for graphs, by default 'dark', only alternative is 'light'.
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

        
    if theme is None:
        theme = _theme

    if theme == 'dark':
        edge_color = '#e6e6e6'
        node_edgecolor = '#888888'
        node_fillcolor = "#c6c6c6"
        start_edgecolor = 'black'
        start_fillcolor = '#777777'
        abs_edgecolor = 'black'
        abs_fillcolor = '#777777'
        bgcolor = '#1F1F1F'
        subgraph_label_fontcolor = '#e6e6e6'
        subgraph_bgcolor='#3F3F3F'
        husl_colors = _get_color(10, lightness=0.7)
    else:
        edge_color = '#009900'
        node_edgecolor='black'
        node_fillcolor='#eeeeee'
        edge_color='black' 
        start_edgecolor='black'
        start_fillcolor='#eeeeee'
        abs_edgecolor='black'
        abs_fillcolor='#eeeeee'
        bgcolor='transparent'
        subgraph_label_fontcolor = 'black'
        subgraph_bgcolor='whitesmoke'
        husl_colors = _get_color(10, lightness=0.4)

    if graph.vertices_length() > max_nodes:
        raise ValueError(f"Graph has too many nodes ({graph.vertices_length()}). Please set max_nodes to a higher value.")

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

    subgraph_attr = dict(rank='same', style='filled', color=subgraph_bgcolor,
                          fontcolor=subgraph_label_fontcolor)
    subgraphs = defaultdict(list)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        if i == 0:
            dot.node(str(vertex.index()), 'S', 
                     style='filled', edge_color=start_edgecolor, fillcolor=start_fillcolor)
        elif not vertex.edges():
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())), 
                     style='filled', edge_color=abs_edgecolor, fillcolor=abs_fillcolor)
        elif subgraphfun is not None:
            subgraphs[f'cluster_{subgraphfun(vertex.state())}'].append(i)
        else:
            dot.node(str(vertex.index()), ','.join(map(str, vertex.state())),
                     style='filled', edge_color=node_edgecolor, fillcolor=node_fillcolor)
    if subgraphfun is not None:
        for sglabel in subgraphs:
            subgraph_attr['label'] = sglabel.replace('cluster_', '')
            with dot.subgraph(name=sglabel, graph_attr=subgraph_attr) as c:
                for i in subgraphs[sglabel]:
                    vertex = graph.vertex_at(i)
                    c.node(str(vertex.index()), ','.join(map(str, vertex.state())))
    return dot
