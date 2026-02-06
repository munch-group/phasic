from __future__ import annotations

import os
import subprocess
import graphviz
from collections import defaultdict
import seaborn as sns
import matplotlib
import matplotlib.colors
from itertools import cycle

from typing import Any, TypeVar
from collections.abc import Callable, Generator

from .logging_config import get_logger
logger = get_logger(__name__)

GraphType = TypeVar('Graph')


def _get_color(n: int, lightness: float = 0.4) -> Generator[str, None, None]:
    """Generate an infinite cycle of hex color strings from a HUSL palette.

    Parameters
    ----------
    n : int
        Number of distinct colors in the palette.
    lightness : float
        Lightness parameter for the HUSL palette, by default 0.4.

    Yields
    ------
    str
        Hex color string (e.g., ``'#a1b2c3'``).
    """
    color_cycle = cycle([matplotlib.colors.to_hex(c) for c in sns.husl_palette(n, l=lightness)])
    for color in color_cycle:
        yield color

def _format_rate(rate: float) -> str:
    """Format a transition rate for display on graph edges.

    Parameters
    ----------
    rate : float
        The transition rate value.

    Returns
    -------
    str
        Formatted string using fixed-point for integers, scientific
        notation otherwise.
    """
    if rate == round(rate):
        return f"{rate:.2f}"
    else:
        return f"{rate:.2e}"



def plot_graph(graph: Any, filename: str | None = None,
               subgraphfun: Callable[..., str] | None = None,
               by_state: Callable[..., str] | None = None,
               by_index: Callable[[int], str] | None = None,
               max_nodes: int = 100,
               dark: bool | None = None,
               constraint: bool = True, ranksep: float = 1, nodesep: float = 1, rankdir: str = "LR",
               size: tuple[int, int] = (7, 7), fontsize: int = 12, rainbow: bool = True, penwidth: float = 1,
               seed: int = 1,
               **kwargs: Any) -> graphviz.Digraph | None:
    """Plot a graph using graphviz.

    Parameters
    ----------
    graph : Graph
        The phasic graph object to visualize.
    filename : str | None
        If provided, save the graph to this file. The file extension
        determines the output format (e.g., ``'graph.pdf'``).
    subgraphfun : Callable[..., str] | None
        Deprecated. Use ``by_state`` instead. Callback function defining
        subgraph clusters by state.
    by_state : Callable[..., str] | None
        Callback function defining subgraph clusters. Takes a state as
        input and returns a string used as the subgraph label.
    by_index : Callable[[int], str] | None
        Callback function defining subgraph clusters. Takes a vertex
        index as input and returns a string used as the subgraph label.
    max_nodes : int
        Maximum number of vertices to plot, by default 100.
    dark : bool | None
        Whether to use dark mode for the graph. Detected automatically
        from the VS Code theme if ``vscodenb`` is available.
    constraint : bool
        Graphviz constraint attribute, by default True.
    ranksep : float
        Graphviz ranksep attribute, by default 1.
    nodesep : float
        Graphviz nodesep attribute, by default 1.
    rankdir : str
        Graphviz rankdir attribute, by default ``"LR"``.
    size : tuple[int, int]
        Graphviz size as ``(width, height)``, by default ``(7, 7)``.
    fontsize : int
        Graphviz fontsize attribute, by default 12.
    rainbow : bool
        Whether to color edges with random colors, by default True.
    penwidth : float
        Graphviz penwidth attribute, by default 1.
    seed : int
        Random seed for graph layout, by default 1.
    **kwargs : Any
        Additional graphviz graph attributes.

    Returns
    -------
    graphviz.Digraph | None
        Graphviz Digraph object for display in Jupyter notebooks,
        or ``None`` if the graph exceeds ``max_nodes``.
    """
    try:
        from vscodenb import is_vscode_dark_theme
        dark = is_vscode_dark_theme()
    except ImportError:
        logger.warning(f"vscodenb is not available. Defaulting to light theme.")
        dark = False

    # always light theme when executing via nbconvert
    if os.environ.get('NBCONVERT', False):
        dark = False

    subprocess.check_call(['dot', '-c']) # register layout engine

    # backwards comp
    if by_state is None and subgraphfun is not None:
        by_state = subgraphfun

    if by_state and by_index:
        assert "Do not use both by_index and by_state"

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
                    c.node(str(vertex.index()))

    if filename:
        name, suffix = filename.rsplit('.', 1)
        dot.render(name, format=suffix, cleanup=True)

    return dot
