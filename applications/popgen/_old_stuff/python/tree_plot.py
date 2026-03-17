from scipy import signal
import numpy as np
import pandas as pd
import ete3
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from itertools import takewhile
%config InlineBackend.figure_formats = ['retina', 'png']

plt.style.use('default')

n = 10000
groups = zip(
    np.random.choice(['I', 'R', 'E'], size=n, replace=True),
    np.random.choice(['1', '2', '3'], size=n, replace=True),
    # np.random.choice(['1b2', '1', ''], size=n, replace=True),
    # np.random.choice(['a', 'b', 'c'], size=n, replace=True),
    np.random.choice(['a1b2', 'a1b1', 'a1', ''], size=n, replace=True),
)
data = pd.DataFrame(dict(haplotype = [''.join(x) for x in groups], 
     case=np.random.randint(0, 2, size=n)
     ))
data.head()



def find_lowest_cell(table):
    x = 1
    y = 0
    min_val = table[x][y]
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i][j] < min_val:
                min_val = table[i][j]
                x = i
                y = j
    return [x, y]

def link(x, y, wx, wy):
    return (x * wx + y * wy) / (wx + wy)

def update_table(table, a, b, weight_a, weight_b):
    for i in range(0, b):
        table[b][i] = link(table[b][i], table[a][i], weight_b, weight_a)
    for j in range(b+1, a):
        table[j][b] = link(table[j][b], table[a][j], weight_b, weight_a)
    for i in range(a+1, len(table)):
        table[i][b] = link(table[i][b], table[i][a], weight_b, weight_a)
    for i in range(a+1, len(table)):
        del table[i][a]
    del table[a] 

def update_labels(labels, i, j, di, dj):
    labels[j] = "({}:{},{}:{})".format(labels[j], dj, labels[i], di)
    del labels[i]

def upgma(mat, names):

    table = mat[:]
    labels = names[:]
    node_heights = [0 for _ in labels]

    while len(labels) > 1:
        i, j = find_lowest_cell(table)
        
        dist = table[i][j]

        wi = max(1, labels[i].count(':'))
        wj = max(1, labels[j].count(':'))

        new_node_height = dist / 2
        di = new_node_height - node_heights[i]
        dj = new_node_height - node_heights[j]
        
        update_table(table, i, j, wi, wj)
        update_labels(labels, i, j, di, dj)
        node_heights[j] = new_node_height
        del node_heights[i]
        
    return labels[0] + ';'

haplotypes = data.haplotype.unique().tolist()

max_len = max(map(len, haplotypes))

lowtri = list()
for i in range(len(haplotypes)):
    lowtri.append([])
    for j in range(i):
        same = 0
        for k, (a, b) in enumerate(zip(haplotypes[i], haplotypes[j])):
            if a != b:
                break
            same += 1
        lowtri[i].append(max_len-same)
        
clustering = upgma(lowtri, haplotypes)
upgma_tree = ete3.Tree(clustering, format=1)

def plot_tree(t, ax, leaf_colors=None, show_inner_nodes=False, fontsize=10, 
              text_offset=None, margins=(0.5, 1, 0.5, 1), align_labels=False): # top, right, bottom, left

    y_offset = len(t.get_leaves())
    for node in t.traverse("preorder"):
        node.x_offset = node.dist + sum(x.dist for x in node.get_ancestors())
        if node.is_leaf():
            y_offset -= 1
            node.y_offset = y_offset

    for node in t.traverse("postorder"):
        if not node.is_leaf():
            node.y_offset = sum(x.y_offset for x in node.children) / len(node.children)

    horizontal_lines = list()
    vertical_lines = list()
    node_coords = list()
    leaf_coords = list()
    texts = list()
    max_x_offset = 0
    for node in t.traverse("postorder"):
        max_x_offset = max(max_x_offset, node.x_offset)
        node_coords.append((node.x_offset, node.y_offset))
        if node.is_leaf():
            leaf_coords.append([node.name, node.x_offset, node.y_offset])
        if not node.is_root():
            y = node.y_offset
            horizontal_lines.append(([node.up.x_offset, node.x_offset], [y, y]))
        if not node.is_leaf():
            c = sorted(node.children, key=lambda x: x.y_offset)
            bottom, top = c[0], c[-1]
            x = node.x_offset
            vertical_lines.append(([x, x],[bottom.y_offset, top.y_offset]))
            y = (bottom.y_offset + top.y_offset) / 2

            strings =[x.name for x in c]
            prefix = ''.join(c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*strings)))

            df = data.loc[data.haplotype.str.startswith(prefix)]
            suffix = f'({df.loc[df.case == 1].index.size}/{df.loc[df.case == 0].index.size})'

            label  = prefix + ' ' + suffix
            node.name = prefix
            texts.append((x, y, label))

    for i in range(len(horizontal_lines)):
        horizontal_lines[i][0][0] -= max_x_offset
        horizontal_lines[i][0][1] -= max_x_offset
    for i in range(len(vertical_lines)):
        vertical_lines[i][0][0] -= max_x_offset
        vertical_lines[i][0][1] -= max_x_offset
    for i in range(len(leaf_coords)):
        leaf_coords[i][1] -= max_x_offset
            
    # draw the tree:
    for x in horizontal_lines:
        ax.plot(*x, c='black', linewidth=0.8)
    for x in vertical_lines:
        ax.plot(*x, c='black', linewidth=0.8)

    for x, y, txt in texts:
        ax.text(x - max_x_offset, y, txt, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', color='red')

    if text_offset is None:
        text_offset = max_x_offset / 20
        
    for name, x, y in leaf_coords:
                
        if align_labels:
            ax.text(0+text_offset, y, name, fontsize=fontsize,
                    verticalalignment='center', horizontalalignment='left')
            if leaf_colors is None:
                color = 'black'
            else:
                color = leaf_colors[name]
            ax.plot(x, y, c=color, marker="o", ms=3)
            ax.add_line(Line2D((x, text_offset), (y, y), linewidth=0.8, color='grey', linestyle='dashed', zorder=0))
        else:
            ax.text(x+text_offset, y, name, fontsize=fontsize,
                    verticalalignment='center', horizontalalignment='left')
            if leaf_colors is None:
                color = 'black'
            else:
                color = leaf_colors[name]
            ax.plot(x, y, c=color, marker="o", ms=3)

    ax.set_xlim(-margins[3]-max_x_offset, margins[1])
    ax.set_ylim(-margins[2], len(leaf_coords)-1+margins[0])

    ax.get_yaxis().set_visible(False)

    ax.spines['top'].set_visible(False) 
    ax.spines['left'].set_visible(False) 
    ax.spines['right'].set_visible(False)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        
    return leaf_coords
 
if __name__ == "__main__":

    fig, ax = plt.subplots(1, 3, figsize=(10, 8), width_ratios=[5, 1, 1], sharey=True)
    leaf_info = plot_tree(upgma_tree, ax[0], text_offset=0.1,margins=(1, 0.1, 1, 0.1), fontsize=7)

    base = 0
    for _ in range(5):
        width = np.random.random(size=len(haplotypes))
        ax[1].barh(width=width, y=list(range(len(haplotypes))))
        base += width

    df = pd.DataFrame({hap: np.random.normal(loc=np.random.random()*10, scale=0.5, size=100) for i, hap in enumerate(haplotypes)}).reset_index().melt(id_vars='index', var_name='haplotype', value_name='value')
    sns.boxplot(data=df, x="value", y="haplotype", ax=ax[2], orient='h')

    plt.tight_layout()

    plt.show()