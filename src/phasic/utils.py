from __future__ import annotations

import numpy as np

from functools import wraps, partial
import time
from pathlib import Path
from IPython.display import display, Markdown, HTML
import base64

from typing import Any
from collections.abc import Callable


def hand_off(target_func: Callable[..., Any]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that forwards all parameters to another function.

    Parameters
    ----------
    target_func : Callable[..., Any]
        The function to forward calls to.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        A decorator that wraps the decorated function to call ``target_func``.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return target_func(*args, **kwargs)
        return wrapper
    return decorator

def download_link(notebook_path: str | Path) -> None:
    """Display an HTML download link for a Jupyter notebook.

    Creates a base64-encoded download link and displays it as right-aligned
    HTML in a Jupyter notebook environment.

    Parameters
    ----------
    notebook_path : str | Path
        Path to the notebook file to create a download link for.
    """
    nb_path = Path(notebook_path)
    b64 = base64.b64encode(nb_path.read_bytes()).decode()
    display(HTML(f'''<div style="text-align:right;">
  <a href="data:application/octet-stream;base64,{b64}" download="{nb_path.name}" title="Download notebook" style="text-decoration:none;">
    Download notebook
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  </a>
</div>'''))


import matplotlib.pyplot as plt
from collections import Counter


class _Node:
    __slots__ = ['left', 'right', 'allele', 'x', 'y', 'mutation']

    def __init__(self, left=None, right=None, allele=None):
        self.left = left
        self.right = right
        self.allele = allele
        self.x = self.y = None
        self.mutation = None

    def is_leaf(self):
        return self.left is None

    def n_leaves(self):
        if self.is_leaf():
            return 1
        return self.left.n_leaves() + self.right.n_leaves()

    def leaves(self):
        if self.is_leaf():
            return [self]
        return self.left.leaves() + self.right.leaves()


def _build(alleles, balanced=True):
    if len(alleles) == 1:
        return _Node(allele=alleles[0])
    if balanced:
        mid = (len(alleles) + 1) // 2
    else:
        mid = np.random.randint(1, len(alleles))
    return _Node(left=_build(alleles[:mid], balanced=balanced),
                 right=_build(alleles[mid:], balanced=balanced))


def _merge(trees):
    if len(trees) == 1:
        return trees[0]
    mid = (len(trees) + 1) // 2
    return _Node(left=_merge(trees[:mid]), right=_merge(trees[mid:]))


def _layout(node, spacing=1.5, seed=42):
    counter = [0]

    def set_x(n):
        if n.is_leaf():
            n.x = counter[0] * spacing
            counter[0] += 1
            return
        set_x(n.left)
        set_x(n.right)
        n.x = (n.left.x + n.right.x) / 2

    def set_y(n):
        if n.is_leaf():
            n.y = 0
            return 0
        lh, rh = set_y(n.left), set_y(n.right)
        n.y = max(lh, rh) + 2 + 0.2 * n.n_leaves() + (1 - np.random.rand()) * min(3, n.n_leaves())

        return n.y

    set_x(node)
    set_y(node)


def _find_parent_y(root, target):
    if root.is_leaf():
        return None
    if root.left is target or root.right is target:
        return root.y
    return _find_parent_y(root.left, target) or \
           _find_parent_y(root.right, target)


def _data_units_per_point(ax, direction='x'):
    """Data-space units per typographic point."""
    fig = ax.get_figure()
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if direction == 'x':
        xlim = ax.get_xlim()
        return (xlim[1] - xlim[0]) / (bbox.width * 72)
    else:
        ylim = ax.get_ylim()
        return (ylim[1] - ylim[0]) / (bbox.height * 72)


def tree_illustration(sequence, explain=True, ax=None, 
                      ton=None, ton_label=True, balanced=True, 
                      figsize=(6, 4), seed=42):
    
    if ton:
        explain = False

    np.random.seed(seed)

    counts = Counter(sequence)
    ancestral = counts.most_common(1)[0][0]

    seen = []
    for ch in sequence:
        if ch not in seen:
            seen.append(ch)
    derived = [a for a in seen if a != ancestral]

    n_anc = counts[ancestral]
    n_der = len(derived)

    anc_groups, remaining = [], n_anc
    for i in range(n_der + 1):
        share = remaining // (n_der + 1 - i)
        anc_groups.append(share)
        remaining -= share

    clade_trees, der_roots = [], []
    for i in range(n_der + 1):
        if anc_groups[i] > 0:
            clade_trees.append(
                _build([ancestral] * anc_groups[i], balanced=balanced))
        if i < n_der:
            t = _build([derived[i]] * counts[derived[i]], balanced=balanced)
            t.mutation = (ancestral, derived[i])
            der_roots.append(t)
            clade_trees.append(t)

    root = _merge(clade_trees) if len(clade_trees) > 1 else clade_trees[0]
    _layout(root, seed=seed)

    make_fig = ax is None
    if make_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # col, lw = '#38bdf8', 1.5
    col, lw = '#14C1E3', 1.5
    # highlight_color = '#FF1B1B'
    highlight_color = '#dc2626'

    def draw(node):
        if node.is_leaf():
            return
        

        ax.plot([node.left.x, node.right.x], [node.y, node.y],
                color=col, lw=lw, solid_capstyle='round')
        for child in (node.left, node.right):
            ax.plot([child.x, child.x], [child.y, node.y],
                    color=highlight_color if child.n_leaves() == ton else col, 
                    lw=lw, solid_capstyle='round')
        draw(node.left)
        draw(node.right)

    draw(root)

    # ax.annotate(f'MRCA: {ancestral}', xy=(root.x, root.y + 0.1),
    #             xytext=(root.x, root.y + 0.7),
    #             ha='center', va='bottom', fontsize=11,
    #             fontweight='bold', color='#333',
    #             arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Set limits before computing arrow size
    xs = [l.x for l in root.leaves()]
    ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
    ax.set_ylim(-2, root.y + 3)
    ax.set_aspect('auto')

    # Arrow length in display points → converted to data units
    ARROW_PT = 20
    dup_x = _data_units_per_point(ax, 'x')
    dup_y = _data_units_per_point(ax, 'y')
    arrow_len = ARROW_PT * dup_x
    gap = 2 * dup_x
    # Full label width: letter + gap + arrow + gap + letter ≈
    label_half_w = arrow_len / 2 + gap + 8 * dup_x  # 8pt for a letter

    for leaf in root.leaves():
        ax.scatter(leaf.x, 0, s=40, color=col, zorder=5)

    if explain:
        for leaf in root.leaves():
            c = highlight_color if leaf.allele != ancestral else '#333'
            ax.text(leaf.x, -8 * dup_y, leaf.allele, ha='center', va='top',
                    fontsize=11, fontweight='bold', color=c)

        ax.plot([root.x, root.x], [root.y, root.y + 7 * dup_y],
                color=col, lw=lw, solid_capstyle='round')    
        ax.text(root.x, root.y + 14 * dup_y, f'MRCA: {ancestral}', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333')

    if ton and ton_label:
        ax.text(root.left.x - 20 * dup_x, root.y, f"{ton}'ton\nbranches", ha='right', va='top',
            fontsize=11, fontweight='bold', color=highlight_color)

    # Collect mutation label info, then resolve horizontal overlaps
    labels = []
    for dn in der_roots:
        py = _find_parent_y(root, dn)
        if py is None:
            continue
        mid_y = (py + dn.y) / 2 + 5 * dup_y # nudge up a bit from the middle
        labels.append((dn, py, mid_y))

    # Sort by x so we can check neighbours
    labels.sort(key=lambda t: t[0].x)

    if explain:

        # Nudge overlapping labels vertically
        for i in range(1, len(labels)):
            dn_prev, _, my_prev = labels[i - 1]
            dn_curr, _, my_curr = labels[i]
            dx = abs(dn_curr.x - dn_prev.x)
            if dx < 2 * label_half_w and abs(my_curr - my_prev) < 12 * dup_y:
                # Shift current one up
                py_curr = labels[i][1]
                new_my = py_curr - 6 * dup_y  # place near parent
                labels[i] = (dn_curr, py_curr, new_my)

        for dn, py, mid_y in labels:
            n = dn.n_leaves()
            fr, to = dn.mutation

            arrow_left = dn.x - arrow_len / 2
            arrow_right = dn.x + arrow_len / 2

            ax.text(arrow_left - gap, mid_y, fr, ha='right', va='center',
                    fontsize=11, fontweight='bold', color='#333')
            ax.annotate('', xy=(arrow_right, mid_y),
                        xytext=(arrow_left, mid_y),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))
            ax.text(arrow_right + gap, mid_y, to, ha='left', va='center',
                    fontsize=11, fontweight='bold', color='#dc2626')

            # (n'ton) always below the arrow
            ax.text(dn.x, mid_y - 10 * dup_y, f"({n}'ton)",
                    ha='center', va='top', fontsize=11, fontweight='bold', color='#555')
            ax.plot(dn.x, mid_y, color='#dc2626', markersize=12, zorder=6)

    ax.axis('off')
    if make_fig:
        fig.tight_layout()

    return fig, ax
