library(ggplot2)
theme_set(theme_bw())

despine <- theme(panel.border = element_blank(), panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
                # text=element_text(size=17)
                # text=element_text(family="Arial")
                ) 

# despine <- theme(panel.border = element_blank(), panel.grid.major = element_blank(),
# panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
#                 text=element_text(family="Arial"))

plot_sfs <- function(graph, rewards) {
    sfs <- sapply(1:(dim(rewards)[1]-1), function(i) expectation(graph, rewards[i,]))
    data.frame(
      ton=seq(1,length(sfs)),  
      brlen=sfs
      ) %>% ggplot(aes(x=ton, y=brlen, fill=ton)) + 
      geom_bar(stat = "identity", width=0.8) + scale_fill_viridis() + 
      despine
}

plot_sfs_dph <- function(graph, rewards, trunc=4) {
    result = data.frame()
    for (i in 1:(nrow(rewards)-1)) {
        x <- seq(from = 0, to = trunc, by = 0.01)
        pdf <- dph(x, reward_transform(graph, rewards[i, ]))
        df <- data.frame(probability = pdf, t=x, ton=i)
        result <- rbind(result, df)
    }
    result %>% ggplot(aes(y=probability, x=t, group=ton, color=ton)) +
        geom_line(linewidth=1) + scale_color_viridis() + despine
}

get_exp_mat <- function(graph, rewards) {
    s <- nrow(rewards)
    exp_mat <- matrix(nrow=s+1,ncol=s+1)
    for (i in 0:s) {
      for (j in 0:s) {
        exp_mat[i+1,j+1] <- expectation(graph, rewards[props_to_index(s, i, j, 1),] + rewards[props_to_index(s, i, j, 2),])
      }
    } 
    return(exp_mat)
}
                                  
plot_exp_mat <- function(exp_mat) {  
    df <- as.data.frame(exp_mat) #%>% gather()
    df <- df %>% rownames_to_column('ton1') %>% gather('ton2', 'value', -c(ton1))

    limit <- max(abs(df$value)) * c(-1, 1)
    
    ggplot(df, aes(ton1, ton2)) +
        geom_tile(aes(fill = value)) + 
        geom_text(aes(label = round(value, 2))) +
    scale_x_discrete(labels= seq(0, nrow(exp_mat))) + 
    scale_y_discrete(labels= seq(0, nrow(exp_mat))) + 
    scale_fill_distiller(palette = 'PiYG',direction = 1,
                        limit=limit) +
    theme_minimal() +
     theme(panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank(), 
            text=element_text(size=17))

}

get_cov_mat <- function(graph, rewards) {
    n <- nrow(rewards)
    cov_mat <- matrix(nrow=n-1,ncol=n-1)
    for (i in 1:(n-1)) {
        for (j in 1:(n-1)) {
            cov_mat[i, j] <- covariance(graph, rewards[i,], rewards[j,])
        }
    }
    return(cov_mat)
}
                  
plot_cov_mat <- function(cov_mat) {
   
    df <- as.data.frame(cov_mat)
    df <- df %>% rownames_to_column('ton1') %>% gather('ton2', 'value', -c(ton1))
    
    ggplot(df, aes(ton1, ton2)) +
        geom_tile(aes(fill = value)) + 
        scale_y_discrete(labels= seq(1, nrow(cov_mat))) + 
        scale_fill_continuous(type = "viridis") + 
        theme_minimal() + 
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), text=element_text(size=17)) 

}

plot_sim <- function(graph)
{
    gam <- graph_as_matrix(graph)
    mat <- gam$SIM
    mat <- matrix(as.integer(mat > 0), dim(mat))
    
    rownames(mat) <- 1:nrow(mat)
    colnames(mat) <- 1:ncol(mat)
    df <- as.data.frame(mat)
    df <- df %>% rownames_to_column('n') %>% gather('m', 'value', -c(n))
    df$n <- as.numeric(df$n)
    df$m <- as.numeric(df$m)
    ggplot(df, aes(n, m)) +
        geom_tile(aes(fill = value), show.legend = FALSE) + 
        scale_fill_gradient(low="white", high="black") +
        theme_minimal() + 
        theme(panel.grid.major = element_blank(), 
              panel.grid.minor = element_blank(), 
              axis.title.x = element_blank(),
              axis.title.y = element_blank(),
              text=element_text(size=17))
}


plot_graph <- function(gam, constrained=TRUE, 
                       subgraphs=FALSE, ranksep=2, nodesep=1,
                       subgraphfun=function(state) paste(state[-length(state)], collapse=""), 
                       size=c(6, 6), fontsize=10, rankdir="LR", align=FALSE, nodecolor='white', rainbow=FALSE, penwidth=1) {


    format_rate <- function(rate) {
        # tol = .Machine$double.eps^0.5
        # if (min(abs(c(rate%%1, rate%%1-1))) < tol) {
        if (rate == round(rate)) {
            return(rate)
        } else {
            return(formatC(rate, format = "e", digits = 2))
        }
    }

    random_color <- function() {
        if (rainbow) {
            return(paste("#", paste0(sample(c(0:9, LETTERS[1:6]), 6, T), collapse = ''), sep=''))
        } else {
            return('#000000')
        }
    }

    # subgraphfun <- function(state) as.character(state[length(state)])

    # subgraphfun <- function(state) paste(state[length(state)], collapse="")
    # subgraphfun <- function(state) paste(state[-length(state)], collapse="")

    # sub_graph <- function(state) as.character(state[-length(state)])
    sub_graphs = list()
    state_classes = list()
    
    if (constrained) {
        constrained <- 'true'
    } else {
        constrained <- 'false'
    }

    edge_templ <- '"FROM" -> "TO" [constraint=true, label="LABEL",labelfloat=false,color="COLOR",fontcolor="COLOR"];'
    subgraph_template <- '
    subgraph cluster_FREQBIN {
        style=filled;
        color=whitesmoke;
        node [style=filled];
        NODES;
        label = "FREQBIN";
    }
    '
    start_name <- 'IPV'
    absorbing_name <- 'Absorb'
    edges <- c()
    # IPV edges
    for (i in 1:length(gam$IPV)) {
        if (gam$IPV[i] > 0) {
            edge <- edge_templ
            edge <- sub('FROM', start_name, edge)
            edge <- sub('TO', paste(gam$states[i,], collapse = ","), edge)
            edge <- sub('LABEL', gam$IPV[i], edge)
            edge <- gsub('COLOR', random_color(), edge)                        
            edges <- c(edges, edge)
        }
    }    
    # Matrix edges
    for (i in 1:(nrow(gam$states))) {
        for (j in 1:nrow(gam$states)) {
            if ((i != j) && (gam$SIM[i, j] > 0)) {
                edge <- edge_templ
                edge <- sub('FROM', paste(gam$states[i,], collapse = ","), edge)
                edge <- sub('TO', paste(gam$states[j,], collapse = ","), edge)
                edge <- sub('LABEL', format_rate(gam$SIM[i, j]), edge)
                edge <- gsub('COLOR', random_color(), edge)
                edges <- c(edges, edge)
            }
        }
    }

    absorb_rates <- -rowSums(gam$SIM)
    for (i in 1:nrow(gam$states)) {
        if (absorb_rates[i] > 0) {
            # edges <- c(edges, paste('"', from, '"', ' -> ', 'Absorb', '[constraint=true, label="', label, '",labelfloat=false]', ';', sep='')) 
            edge <- edge_templ
            edge <- sub('FROM', paste(gam$states[i,], collapse = ","), edge)
            edge <- sub('TO', absorbing_name, edge)
            edge <- sub('LABEL', absorb_rates[i], edge)
            edge <- gsub('COLOR', random_color(), edge)            
            edges <- c(edges, edge)
        }
    }

    graph_spec <- paste(edges, collapse = '\n')

    rank_same <- ''

    if (subgraphs) {        
        for (i in 1:(nrow(gam$states))) {
            sg <- subgraphfun(gam$states[i,])
            sub_graphs[[sg]] <- c(sub_graphs[[sg]], paste(gam$states[i,], collapse = ","))
        }
        for (sg in labels(sub_graphs)) {
            
            nodes <- sub_graphs[[sg]]
            tmpl <- subgraph_template
            node_str <- ''
            for (i in 1:length(nodes)) {
                node_str <- paste(node_str, paste('"', nodes[i], '" ', sep=''), sep=' ')
            }
            tmpl <- sub('NODES', node_str, tmpl)
            tmpl <- sub('FREQBIN', sg, tmpl)            
            tmpl <- sub('FREQBIN', sg, tmpl)            
            graph_spec <- paste(graph_spec, tmpl)
        }


        if (align) {
            for (i in 1:(nrow(gam$states))) {
                sc <- paste(head(gam$states[i,], -1), collapse = ",")
                state_classes[[sc]] <- c(state_classes[[sc]], paste(gam$states[i,], collapse = ","))
            }
            for (sc in labels(state_classes)) {
                rank_same <- paste(rank_same, '{rank=same; ', sep='')
                nodes <- state_classes[[sc]]
                for (i in 1:length(nodes)) {
                    rank_same <- paste(rank_same, paste('"', nodes[i], '" ', sep=''), sep=' ')
                }            
                rank_same <- paste(rank_same, ' }', sep='\n')
            }
        }
    
    }

#     style_str <- '
#         rankdir=RANKDIR;
#         size="SIZEX,SIZEY";
#         fontname="Helvetica,Arial,sans-serif"
#     	node [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE]
#     	edge [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE]
#         Absorb [style=filled,color="lightgrey"]
#         IPV [style=filled,color="lightgrey"]
#     '
    
    style_str <- '
        graph [compound=true newrank=true pad="0.5", ranksep="RANKSEP", nodesep="NODESEP"] 
        rankdir=RANKDIR;
        size="SIZEX,SIZEY";
        fontname="Helvetica,Arial,sans-serif"
    	node [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE, style=filled, fillcolor="NODECOLOR"]
    	edge [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE, penwidth=PENWIDTH]
        Absorb [style=filled,color="lightgrey"]
        IPV [style=filled,color="lightgrey"]
        RANKSAME
    '
    style_str <- sub('SIZEX', size[1], style_str)
    style_str <- sub('SIZEY', size[2], style_str)
    style_str <- gsub('FONTSIZE', fontsize, style_str)    
    style_str <- gsub('RANKDIR', rankdir, style_str)    
    style_str <- gsub('RANKSAME', rank_same, style_str)
    style_str <- gsub('RANKSEP', ranksep, style_str)
    style_str <- gsub('NODESEP', nodesep, style_str)
    graph_string <- paste('digraph G {', style_str, graph_spec, '}', sep='\n')
    graph_string <- gsub('NODECOLOR', nodecolor, graph_string)  
    graph_string <- gsub('PENWIDTH', penwidth, graph_string)  
    system("dot -Tsvg -o tmp.svg", input=graph_string, intern=TRUE)
    return(display_svg(file="tmp.svg"))
    }
                  

# plot_graph <- function(gam, constrained=TRUE, size=c(6, 6), fontsize=10, rankdir="LR") {

#     if (constrained) {
#         constrained <- 'true'
#     } else {
#         constrained <- 'false'
#     }

#     edge_templ <- '"FROM" -> "TO" [constraint=true, label="LABEL",labelfloat=false];'
#     start_name <- 'IPV'
#     absorbing_name <- 'Absorb'
#     edges <- c()
#     for (i in 1:length(gam$IPV)) {
#         if (gam$IPV[i] > 0) {
#             edge <- edge_templ
#             edge <- sub('FROM', start_name, edge)
#             edge <- sub('TO', paste(gam$states[i,], collapse = ","), edge)
#             edge <- sub('LABEL', gam$SIM[i, 1], edge)
#             edges <- c(edges, edge)
#         }
#     }    
#     for (i in 1:nrow(gam$states)) {
#         for (j in 1:nrow(gam$states)) {
#             if ((i != j) && (gam$SIM[i, j] > 0)) {
#                 edge <- edge_templ
#                 edge <- sub('FROM', paste(gam$states[i,], collapse = ","), edge)
#                 edge <- sub('TO', paste(gam$states[j,], collapse = ","), edge)
#                 edge <- sub('LABEL', gam$SIM[i, j], edge)
#                 edges <- c(edges, edge)
#             }
#         }
#     }
#     absorb_rates <- -rowSums(gam$SIM)
#     for (i in 1:nrow(gam$states)) {
#         if (absorb_rates[i] > 0) {
#             # edges <- c(edges, paste('"', from, '"', ' -> ', 'Absorb', '[constraint=true, label="', label, '",labelfloat=false]', ';', sep='')) 
#             edge <- edge_templ
#             edge <- sub('FROM', paste(gam$states[i,], collapse = ","), edge)
#             edge <- sub('TO', absorbing_name, edge)
#             edge <- sub('LABEL', absorb_rates[i], edge)
#             edges <- c(edges, edge)
#         }
#     }
#     graph_spec <- paste(edges, sep='\n', collapse = '')
#     style_str <- '
#         rankdir=RANKDIR;
#         size="SIZEX,SIZEY";
#         fontname="Helvetica,Arial,sans-serif"
#     	node [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE]
#     	edge [fontname="Helvetica,Arial,sans-serif", fontsize=FONTSIZE]
#         Absorb [style=filled,color="lightgrey"]
#         IPV [style=filled,color="lightgrey"]
#     '
#     style_str <- sub('SIZEX', size[1], style_str)
#     style_str <- sub('SIZEY', size[2], style_str)
#     style_str <- gsub('FONTSIZE', fontsize, style_str)    
#     style_str <- gsub('RANKDIR', rankdir, style_str)    
#     graph_string <- paste('digraph G {', style_str, graph_spec, '}', sep='\n')
#     system("dot -Tsvg -o tmp.svg", input=graph_string, intern=TRUE)
#     return(display_svg(file="tmp.svg"))
#     }