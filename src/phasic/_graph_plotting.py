"""Plotting / visualization functions for :class:`~phasic.Graph`.

Extracted verbatim from ``Graph`` (Stage-3 WS-C). Pure relocation: the bodies
are unchanged and use lazy local imports. These are assigned onto ``Graph`` as
class attributes in ``__init__.py`` so they stay DIRECT members of Graph — that
keeps quartodoc documenting them on the Graph page (include_inherited=false).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple


def plot(
    graph: Any, 
    filename: str | None = None,
    wrap: bool|int = True,
    label_fmt: Callable[[Any], str] | None = None,
    rate_fmt: Callable[[Any], str] | None = None,
    color_by: Sequence[float] | None = None,
    color_by_cmap: str = 'viridis',
    color_by_lim: Tuple[float] | None = None,
    color_by_alpha: float | None = 1.0,
    subgraphfun: Callable[..., str] | None = None,
    by_state: Callable[..., str] | None = None,
    by_index: Callable[[int], str] | None = None,
    max_nodes: int = 100,
    dark: bool | None = None,
    constraint: bool = True, ranksep: float = 1, nodesep: float = 1, rankdir: str = "LR",
    size: tuple[int, int] = (7, 7), fontsize: int = 12, rainbow: bool = True, penwidth: float = 1,
    taillabel : bool = False,
    seed: int = 1,
    graph_attr: dict = {},
    node_attr: dict = {},
    edge_attr: dict = {}
    ) -> graphviz.Digraph | None:
    """Plot a graph using graphviz.

    Parameters
    ----------
    graph : Graph
        The phasic graph object to visualize.
    filename : str | None
        If provided, save the graph to this file. The file extension
        determines the output format (e.g., ``'graph.pdf'``).
    wrap : bool | int
        Whether to wrap vertex labels, and if so, the maximum number of
        characters per line. By default True.
    label_fmt : Callable[..., str] | None
        Callable for format node labels:
    rate_fmt : Callable[float] | None
        Callable for format edge labels:
    color_by : List[float] | None
        List of values used for node fill colors.
    color_by_cmap : List[float] | None
        List of values used for node fill colors.
    color_by_lim: Tuple[float] | None
        Color map min, max limits for use with color_by. Default is
        min and max of values given by color_by.
    color_by_alpha: float | None
        Alpha value for node fill color with color_by.
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
    taillabel : bool
        Use taillabel instead of xlabel, by default False
    seed : int
        Random seed for graph layout, by default 1.
    graph_attr : dict
        graphviz graph attributes to override defaults.
    node_attr : dict
        graphviz node attributes to override defaults.
    edge_attr : dict
        graphviz edge attributes to override defaults.

    Returns
    -------
    graphviz.Digraph | None
        Graphviz Digraph object for display in Jupyter notebooks,
        or ``None`` if the graph exceeds ``max_nodes``.
    """
    import math
    import os
    import subprocess
    import graphviz
    from collections import defaultdict
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from itertools import cycle
    from functools import partial

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
            Formatted string using fixed-point for integers, float/scientific
            notation otherwise.
        """
        if rate == round(rate):
            return f"{rate}"
        else:
            return f"{rate:.2e}"

    def _values_to_hex(values, palette_name, vmin=None, vmax=None, alpha=None):
        values = np.asarray(values, dtype=float)
        cmap = matplotlib.colormaps[palette_name]
        norm = matplotlib.colors.Normalize(
            vmin=vmin if vmin is not None else np.nanmin(values),
            vmax=vmax if vmax is not None else np.nanmax(values),
        )
        rgba = cmap(norm(values), alpha=alpha)
        return [matplotlib.colors.to_hex(c, keep_alpha=alpha is not None) for c in rgba]

    def format_label(vertex, wrap=True, max_cols=8):
        state = vertex.state()
        n = len(state) 
        if wrap is False or n <= max_cols:
            return ','.join(map(str, state))
        
        if wrap is True:


            best = None
            for c in range(1, n + 1):
                rows = -(-n // c)                 # ceil(n / c)
                last = n - (rows - 1) * c         # items in last row, in [1, c]
                if 2 * last < c:                  # last row must be at least half full
                    continue
                score = abs(c - 2 * rows)         # cols closest to twice rows
                if best is None or score < best[0]:
                    best = (score, rows, c)
            _, rows, cols = best
        elif isinstance(wrap, int):
            cols = wrap
        else:
            cols = 9999
        l = []
        for i in range(1+n//cols):
            r = ','.join(map(str, state[i*cols:(i+1)*cols]))
            if not r:
                break
            l.append(r)
        return ',\n'.join(l)


    try:
        from vscodenb import is_vscode_dark_theme
        dark, bg_color = is_vscode_dark_theme()
    except ImportError:
        logger.warning(f"vscodenb is not available. Defaulting to light theme.")
        dark, bg_color = (False, 'white')

    # always light theme when executing via nbconvert
    if 'NBCONVERT_BGCOLOR' in os.environ:
        dark, bg_color = (False, os.environ['NBCONVERT_BGCOLOR'])

    if label_fmt is None:
        label_fmt = partial(format_label, wrap=wrap)
    elif label_fmt is False:
        label_fmt = lambda vertex: ''

    if rate_fmt is None:
        rate_fmt = _format_rate
    elif rate_fmt is False:
        rate_fmt = lambda x: ''

    subprocess.check_call(['dot', '-c']) # register layout engine

    # backwards comp
    if by_state is None and subgraphfun is not None:
        by_state = subgraphfun

    if by_state and by_index:
        assert "Do not use both by_index and by_state"

    # get matplotlib background color for graph background
    plt.ioff()
    fig, ax = plt.subplots()
    bg_color = ax.get_facecolor()
    plt.close(fig)
    plt.ion()
    # if sum(bg_color) == 0: # black
    #     bg_color = '#1F1F1F'
    # else:
    bg_color = matplotlib.colors.to_hex(bg_color)
    # if dark:
    #     bg_color = '#1F1F1F'


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
        # bgcolor = '#1F1F1F'
        bgcolor = bg_color
        fontcolor = 'black'
        subgraph_label_fontcolor = '#e6e6e6'
        subgraph_bgcolor='#2e2e2e'
        subgraph_edgecolor='#e6e6e6'
        husl_colors = _get_color(10, lightness=0.7)
    else:
        edge_color = '#3e3e3e'
        node_edgecolor='black'
        node_fillcolor='#eeeeee'
        # edge_color='black'
        start_edgecolor='black'
        start_fillcolor='#bbbbbb'
        abs_edgecolor='black'
        abs_fillcolor='#bbbbbb'
        aux_edgecolor='black'
        aux_fillcolor='#bbbbbb'
        # bgcolor='transparent'
        bgcolor=bg_color
        fontcolor = 'black'
        subgraph_label_fontcolor = 'black'
        # subgraph_bgcolor='white'
        subgraph_bgcolor=bg_color
        subgraph_edgecolor='black'
        husl_colors = _get_color(10, lightness=0.5)

    node_fill_colors = {}
    if color_by:
        if len(color_by) != graph.vertices_length():
            raise ValueError('List of colors passed to color_by must match nr of vertices in graph.')
        vmin, vmax = min(color_by), max(color_by)
        if color_by_lim and color_by_lim[0] is not None:
            vmin = color_by_lim[0]
        if color_by_lim and color_by_lim[1] is not None:
            vmax = color_by_lim[1]
        node_fill_colors = _values_to_hex(color_by, color_by_cmap, vmin=vmin, vmax=vmax, alpha=color_by_alpha)
    
    if graph.vertices_length() > max_nodes:
        print(f"Graph has too many nodes ({graph.vertices_length()}). Please set max_nodes to a higher value.")
        return None

    _graph_attr = dict(compound='true', newrank='true', pad='0.5',
                    ranksep=str(ranksep), nodesep=str(nodesep),
                    bgcolor=bgcolor, rankdir=rankdir, ratio="auto",
                    size=f'{size[0]},{size[1]}',
                    start=str(seed),
                    fontname="Helvetica,Arial,sans-serif")
    _node_attr = dict(style='filled', color='black',
                    fontname="Helvetica,Arial,sans-serif",
                    fontsize=str(fontsize),
                    fillcolor=str(node_fillcolor))
    _edge_attr = dict(constraint='true' if constraint else 'false',
                    style='filled', labelfloat='false', labeldistance='0',
                    fontname="Helvetica,Arial,sans-serif",
                    color=edge_color,
                    fontsize=str(fontsize), penwidth=str(penwidth))
    
    _graph_attr.update(graph_attr)
    _node_attr.update(node_attr)
    _edge_attr.update(edge_attr)

    _graph_attr = dict((k, str(v)) for k, v in _graph_attr.items())
    _node_attr = dict((k, str(v)) for k, v in _node_attr.items())
    _edge_attr = dict((k, str(v)) for k, v in _edge_attr.items())

    dot = graphviz.Digraph(graph_attr=_graph_attr, node_attr=_node_attr, edge_attr=_edge_attr)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        for edge in vertex.edges():
            # if 'color' in edge_attr:
            #     color = edge_attr['color']
            # elif rainbow:
            #     color = next(husl_colors)
            # else:
            #     color = edge_color
            if rainbow:
                # color = next(husl_colors)
                _edge_attr['color'] = next(husl_colors)

            if taillabel:
                _edge_attr['taillabel'] = rate_fmt(edge.weight())                    
            else:
                _edge_attr['xlabel'] = rate_fmt(edge.weight())

            dot.edge(str(vertex.index()), str(edge.to().index()),
                fontcolor=edge_attr.get('labelfontcolor', _edge_attr['color']),
                # xlabel=rate_fmt(edge.weight()), 
                # taillabel=rate_fmt(edge.weight()), 
                **_edge_attr
                )
        #   if rainbow:
        #         _edge_attr['color'] = next(husl_colors)
        #     dot.edge(str(vertex.index()), str(edge.to().index()),                         
        #         xlabel=rate_fmt(edge.weight()), 
        #         # fontcolor=edge_attr.get('labelfontcolor', color),
        #         **dict(_edge_attr, 
        #                     color=edge_attr.get('color', edge_color), 
        #                     ),
        #         )                


    subgraph_attr = dict(rank='same',
                        style='filled',
                        fillcolor=subgraph_bgcolor,
                        color=subgraph_edgecolor,
                        fontcolor=subgraph_label_fontcolor)
    subgraphs = defaultdict(list)
    for i in range(graph.vertices_length()):
        vertex = graph.vertex_at(i)
        label = label_fmt(vertex)
        if i == 0:
            dot.node(str(vertex.index()), 'S',
                     **dict(_node_attr, 
                            edge_color=node_attr.get('edge_color', start_edgecolor), 
                            fillcolor=node_attr.get('fillcolor', start_fillcolor)
                            ),
                     )
        elif not vertex.state().sum() and vertex.rate() == 1 and len(vertex.edges()) == 1:
            dot.node(str(vertex.index()), 'AUX',
                     **dict(_node_attr, 
                            edge_color=node_attr.get('edge_color', aux_edgecolor), 
                            fillcolor=node_attr.get('fillcolor', aux_fillcolor)
                            ),
                    # style='filled', edge_color=aux_edgecolor, fillcolor=aux_fillcolor
                    )
        elif not vertex.edges():
            dot.node(str(vertex.index()), label,
                     **dict(_node_attr, 
                            edge_color=node_attr.get('edge_color', abs_edgecolor), 
                            fillcolor=node_attr.get('fillcolor', abs_fillcolor)
                            ),
                    # style='filled', edge_color=abs_edgecolor, fillcolor=abs_fillcolor
                    )
        else:
            if node_fill_colors:
                node_fill = node_fill_colors[i]
                fontcolor = node_attr.get('fontcolor', fontcolor)
                # luminance = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(node_fill))[2]
                try:
                    import colorsys
                    lightness = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(node_fill))[1]
                    _font_color = 'white' if lightness < 0.5 else fontcolor
                except ImportError:
                    _font_color = fontcolor
            else:
                node_fill = node_attr.get('fillcolor', node_fillcolor)
                _font_color = node_attr.get('fontcolor', fontcolor)
            dot.node(str(vertex.index()), label,
                     **dict(_node_attr, 
                            edge_color=node_attr.get('edge_color', edge_color), 
                            fillcolor=node_fill, fontcolor=_font_color
                            ),                         
                    # style='filled', edge_color=node_edgecolor, fillcolor=node_fillcolor
                    )

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


def plot_scc_decomp(self,
                            figsize: tuple[float, float] | None = None,
                            cmap: str = 'viridis',
                            show_indices: bool = True,
                            annotate_sizes: bool = True,
                            title: bool = False,
                            ax: Any = None) -> Any:
    """Visualise the SCC decomposition of this graph as a
    level-wise treemap.

    Rows correspond to the levels of the SCC condensation.
    The source-side (start vertex, where the chain enters) is
    drawn at the **top** of the figure; the sink-side
    (absorbing state) is at the **bottom** — time flows
    downward.

    Within a row, each tile is one SCC, with width
    proportional to the SCC's vertex count, drawn at a
    common absolute scale shared across all rows. So a
    narrow row really has fewer total vertices than a wide
    one. SCCs at the same level are eliminated independently
    when ``parallel_elimination=True`` is enabled, so wide
    rows signal good parallelism potential and narrow rows
    are elimination bottlenecks.

    Level labels on the left margin look like ``L7 (16)`` —
    the level number followed by the count of parallel SCCs
    at that level. Note that the C-side composer processes
    levels in the opposite of the plot's vertical order
    (sink-first, bottom-up), but that detail does not
    affect interpretation: parallelism is per-row in either
    direction.

    This is a structural visualisation only — it does not
    depend on any runtime telemetry. To assess actual cache
    hit/miss behaviour after a compose, use
    ``phasic.cache.scc_compose_stats()``.

    Parameters
    ----------
    figsize : tuple of float
        Matplotlib figure size in inches. Ignored if ``ax``
        is provided.
    cmap : str
        Matplotlib colormap name. Tiles are coloured by SCC
        index to make adjacent SCCs visually distinct.
    show_indices : bool
        Print the SCC index inside each tile when the tile
        is wide enough.
    annotate_sizes : bool
        Print the vertex count alongside the index.
    title : bool
        Whether to show a title. Default is False. If True adds a one-line summary of the
        decomposition (number of SCCs, levels, widest row).
    ax : matplotlib.axes.Axes or None
        Existing axes to draw into. If ``None``, a new figure
        is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the treemap was drawn into.

    Examples
    --------
    >>> import phasic
    >>> g = phasic.Graph(my_callback)
    >>> ax = g.plot_scc_decomp()
    >>> ax.figure.savefig('scc.pdf')

    See Also
    --------
    scc_decomposition : underlying SCC structure
    phasic.distributed_scc.compute_scc_levels : level grouping
        used by this plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from phasic.distributed_scc import compute_scc_levels

    scc_graph = self.scc_decomposition()
    n_sccs = len(scc_graph)
    if n_sccs == 0:
        raise ValueError(
            "Graph has no SCCs to plot (empty decomposition).")

    sizes = [scc_graph.scc_at(i).size() for i in range(n_sccs)]
    levels = compute_scc_levels(scc_graph)  # sink-first
    widest = max(len(lvl) for lvl in levels)
    total_vertices = sum(sizes)

    # Each row has its own horizontal scale (so tiles fill
    # the row width). Tile widths within a row are
    # proportional to SCC vertex count.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colours = plt.get_cmap(cmap)
    # Normalise colour by SCC index so adjacent SCCs differ.
    def _colour(idx: int):
        return colours((idx % max(n_sccs, 1)) / max(n_sccs - 1, 1))

    n_levels = len(levels)
    row_height = 1.0

    # Absolute scale: 1 horizontal unit = 1 vertex. Each
    # tile is exactly `sizes[i]` units wide. Rows with fewer
    # total vertices look proportionally narrower than rows
    # with more total vertices — that's the whole point of
    # "block width = vertex count".
    gap = 0.4  # absolute horizontal gap between tiles, in vertex units
    max_row_width = max(
        sum(sizes[i] for i in lvl) + max(0, len(lvl) - 1) * gap
        for lvl in levels
        if lvl
    )

    for row_idx, level_sccs in enumerate(levels):
        # Source at top, sink at bottom — time flows downward.
        # `levels` is sink-first (level 0 = sinks), so the
        # last level (the source / start vertex) goes at the
        # top of the figure.
        y = row_idx * row_height
        if not level_sccs:
            continue
        # Centre the row inside [0, max_row_width].
        row_width = (sum(sizes[i] for i in level_sccs)
                     + (len(level_sccs) - 1) * gap)
        x = (max_row_width - row_width) / 2.0
        for i in level_sccs:
            w = sizes[i]
            rect = mpatches.Rectangle(
                (x, y + 0.05), w, row_height - 0.1,
                facecolor=_colour(i), edgecolor='black',
                linewidth=0.5)
            ax.add_patch(rect)
            # Label if tile is wide enough relative to the
            # whole figure (use absolute units now).
            if show_indices and w / max_row_width > 0.03:
                if annotate_sizes:
                    label = f"#{i}\n{sizes[i]}v"
                else:
                    label = f"#{i}"
                ax.text(x + w / 2, y + row_height / 2, label,
                        ha='center', va='center', fontsize=8,
                        color='white' if w / max_row_width > 0.05 else 'black')
            x += w + gap

        # Level label on the left margin.
        ax.text(-0.01 * max_row_width, y + row_height / 2,
                f"L{row_idx} ({len(level_sccs)})",
                ha='right', va='center', fontsize=9,
                family='monospace')

    ax.set_xlim(-0.15 * max_row_width, max_row_width * 1.02)
    ax.set_ylim(-0.05, n_levels * row_height + 0.05)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # title=True -> auto-summary (as documented); a string -> verbatim;
    # False/None -> no title. (Previously checked `is None`, which never
    # fired once the default became the bool False, so title=True wrongly
    # set the literal "True".)
    if title is True:
        title = (f"{n_sccs} SCCs across "
                 f"{n_levels} levels. widest {widest}, "
                 f"{total_vertices} vertices total.")
    if title:
        ax.set_title(title)

    return ax
