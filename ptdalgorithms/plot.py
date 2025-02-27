from IPython.display import Image
from graphviz import Digraph

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx

def plot_graph(gam, constraint=True, subgraphs=False, ranksep=2, nodesep=1, splines=True, subgraphfun=lambda state, index: ','.join(map(str, state[:-1])), size=(6, 6), fontsize=10, rankdir="LR", align=False, nodecolor='white', rainbow=False, penwidth=1):
    G = nx.DiGraph()
    for i in range(gam['states'].shape[0]):
        G.add_node(i, label=','.join(map(str, gam['states'][i, :])))
    for i in range(gam['states'].shape[0]):
        for j in range(gam['states'].shape[0]):
            if i != j and gam['SIM'][i, j] > 0:
                G.add_edge(i, j, label=format_rate(gam['SIM'][i, j]), color=random_color(rainbow))
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color=nodecolor, font_size=fontsize, node_size=500, edge_color=[G[u][v]['color'] for u, v in G.edges()])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=fontsize)
    plt.show()

def format_rate(rate):
    if rate == round(rate):
        return rate
    else:
        return f"{rate:.2e}"

def random_color(rainbow):
    if rainbow:
        return f"#{''.join(np.random.choice(list('0123456789ABCDEF'), 6))}"
    else:
        return '#000000'
    

def plot_graph(graph, 
        constraint=True,
        subgraphs=False, ranksep=2, nodesep=1, splines=True,
        subgraphfun=lambda state, index: ''.join(state[:len(state)]), 
        size = (6, 6), fontsize=10, rankdir="LR", align=False,
        nodecolor='white', rainbow=False, penwidth=1):


    def format_rate(rate):
        # tol = .Machine$double.eps^0.5
        # if (min(abs(c(rate%%1, rate%%1-1))) < tol) {
        if (rate == round(rate)) {
            return(rate)
        } else {
            return(formatC(rate, format = "e", digits = 2))
        }
    }

    random_color = function() {
        if (rainbow) {
            return(paste("#", paste0(sample(c(0:9, LETTERS[1:6]), 6, T), collapse = ''), sep=''))
        } else {
            return('#000000')
        }
    }

    sub_graphs = list()
    state_classes = list()
    
    if (constraint) {
        constraint = 'true'
    } else {
        constraint = 'false'
    }
    
    if (splines == TRUE) {
        splines = 'true'
    } 
    if (splines == FALSE) {
        splines = 'false'
    }

    states = c()
    for (i in 1:(nrow(gam$states))) {
        states = c(states, paste0(i, ' [label="', paste(gam$states[i,], collapse = ","), '"];'))
    }
    
    # edge_templ = '"FROM" -> "TO" [constraint=CONSTRAINT, label="LABEL", labelfloat=false, color="COLOR", fontcolor="COLOR"];'
    edge_templ = '"FROM" -> "TO" [constraint=CONSTRAINT, xlabel="LABEL", labelfloat=false, color="COLOR", fontcolor="COLOR"];'

    # , label2node=true labelOverlay="75%"
    
    subgraph_template = '
    subgraph cluster_FREQBIN {
        rank=same;
        style=filled;
        color=whitesmoke;
        node [style=filled];
        NODES;
        label = "FREQBIN";
    }
    '
    start_name = 'IPV'
    absorbing_name = 'Absorb'
    edges = c()
    # IPV edges
    for (i in 1:length(gam$IPV)) {
        if (gam$IPV[i] > 0) {
            edge = edge_templ
            edge = sub('FROM', start_name, edge)
            edge = sub('TO', i, edge)
            edge = sub('LABEL', gam$IPV[i], edge)
            edge = gsub('COLOR', random_color(), edge)                        
            edges = c(edges, edge)
        }
    }    
    # Matrix edges
    for (i in 1:(nrow(gam$states))) {
        for (j in 1:nrow(gam$states)) {
            if ((i != j) && (gam$SIM[i, j] > 0)) {
                edge = edge_templ
                edge = sub('FROM', i, edge)
                edge = sub('TO', j, edge)
                edge = sub('LABEL', format_rate(gam$SIM[i, j]), edge)
                edge = gsub('COLOR', random_color(), edge)
                edges = c(edges, edge)
            }
        }
    }

    absorb_rates = -rowSums(gam$SIM)
    for (i in 1:nrow(gam$states)) {

        # TODO: Avoid the hack below by changing the function to use the graph instead of the matrix
        if (absorb_rates[i] > abs(1e-14)) {
        # if (absorb_rates[i] > 0) {
            edge = edge_templ
            edge = sub('FROM', i, edge)
            edge = sub('TO', absorbing_name, edge)
            edge = sub('LABEL', absorb_rates[i], edge)
            edge = gsub('COLOR', random_color(), edge)            
            edges = c(edges, edge)
        }
    }

    graph_spec = paste(c(states, edges), collapse = '\n')

    rank_same = ''

    if (subgraphs) {        
        for (i in 1:(nrow(gam$states))) {
            sg = subgraphfun(gam$states[i,], index=i)
            sub_graphs[[sg]] = c(sub_graphs[[sg]], i)
        }
        for (sg in labels(sub_graphs)) {
            
            nodes = sub_graphs[[sg]]
            tmpl = subgraph_template
            node_str = ''
            for (i in 1:length(nodes)) {
                node_str = paste(node_str, paste('"', nodes[i], '" ', sep=''), sep=' ')
            }
            tmpl = sub('NODES', node_str, tmpl)
            tmpl = sub('FREQBIN', sg, tmpl)            
            tmpl = sub('FREQBIN', sg, tmpl)            
            graph_spec = paste(graph_spec, tmpl)
        }


        if (align) {
            for (i in 1:(nrow(gam$states))) {
                sc = paste(head(gam$states[i,], -1), collapse = ",")
                state_classes[[sc]] = c(state_classes[[sc]], i)
            }
            for (sc in labels(state_classes)) {
                rank_same = paste(rank_same, '{rank=same; ', sep='')
                nodes = state_classes[[sc]]
                for (i in 1:length(nodes)) {
                    rank_same = paste(rank_same, paste('"', nodes[i], '" ', sep=''), sep=' ')
                }            
                rank_same = paste(rank_same, ' }', sep='\n')
            }
        }
    
    }

    style_str = '
        graph [compound=true newrank=true pad="0.5", ranksep="RANKSEP", nodesep="NODESEP"] 
        bgcolor=transparent;
        rankdir=RANKDIR;
        splines=SPLINES;
        size="SIZEX,SIZEY";
        fontname="Helvetica,Arial,sans-serif"
    	node [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE, style=filled, fillcolor="NODECOLOR"]
    	edge [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE, penwidth=PENWIDTH]
        Absorb [style=filled,color="lightgrey"]
        IPV [style=filled,color="lightgrey"]
        RANKSAME
    '
    style_str = sub('SIZEX', size[1], style_str)
    style_str = sub('SIZEY', size[2], style_str)
    style_str = gsub('FONTSIZE', fontsize, style_str)    
    style_str = gsub('RANKDIR', rankdir, style_str)    
    style_str = gsub('SPLINES', splines, style_str)    
    style_str = gsub('RANKSAME', rank_same, style_str)
    style_str = gsub('RANKSEP', ranksep, style_str)
    style_str = gsub('NODESEP', nodesep, style_str)
    
    graph_string = paste('digraph G {', style_str, graph_spec, '}', sep='\n')
    graph_string = gsub('NODECOLOR', nodecolor, graph_string)  
    graph_string = gsub('PENWIDTH', penwidth, graph_string)  
    graph_string = gsub('CONSTRAINT', constraint, graph_string)    
    
    system("dot -Tsvg -o tmp.svg", input=graph_string, intern=TRUE)
    return(display_svg(file="tmp.svg"))
}
   