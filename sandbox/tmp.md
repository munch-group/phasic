
I have been thinking more about it. Here is what I think should work. 

1. Partition graph into enhanced subgraphs. Each subgraph is an SCC including the edges connecting other SCCs and the connecting vertices in those SCCs. We denote these "foreign scc" vertices upstream or downstream vertices reflecting whether they connect to upstream or downstream SCCs. We denote the vertices of the current SCC connecting with these vertices for upstream-connecting and downstream-connecting vertices. 
2. Upstream and downstream vertices are included to handle boundary effects when computing the trace. They are cached along with the SCC.
3. Upstream vertices may have edges to other vertices in their upstream SCC, but in the subgraph they are represented without these and their edges are renormalized to sum to 1. These vertices serve as "fake" starting vertices and should, like true starting vertices, not be part of the elimination. 
4. Downstream vertices are "fake" absorbing vertices and should be treated just like real absorbing vertices.
5. Subgraph vertices are indexed so that upstream vertices have lowest indices, then upstream-connecting vertices, then internal SCC vertices, then downstream-connecting vertices, and finally downstream vertices with highest indices: {*upstream, *upstream-connecting, *internal, *downstream-connecting, *downstream}. The * notation is the python-style syntax for sublists of vertices.
6. Create a mapping between graph and subgraph indices. 
7. The graph is split into SCCs and enhanced subgraphs are created.
8. The cache is searched for each subgraph. if not found it is computed and cached.
9. Trace on a subgraph is computed using Algorithm 3 in the paper.
10. The graph is iteratively stitched together from SCC subgraphs upstream to downstream. 
11. Fake starting vertices and their edges are removed. The real starting vertex is not removed.
12. The first subgraph is {Starting_vertex, *downstream}.
13. When connecting an upstream subgraph to a downstream subgraph: 
    - Vertices with the same state vector found in both *upstream vertices in the downstream subgraph and *downstream vertices of the upstream graph, we call sister vertices. Each sister in the downstream subgraph is removed and its edges are attrached to its sister in the upstream subbgraph. 

Try with this example graph
Starting->A, A->B, B->A, B->C, B->E, C->D, C->F, D->E, D->F, E->C




#  First SCC Subgraph (Special Case)

The first SCC in topological order contains the actual starting vertex of the full graph.

Structure:
- Auto-starting vertex (index 0) = actual starting vertex of original graph
- Remaining vertices = downstream vertices that the starting vertex connects to
- NO upstream vertices (this is the first SCC!)

Vertex categories:
- upstream = [] (no upstream - this is first!)
- upstream_connecting = [] (no upstream vertices to connect from)
- internal = [starting_vertex] (just the starting vertex itself)
- downstream_connecting = [] (starting vertex connects to downstream)
- downstream = [vertices that starting_vertex connects to]

#  Other SCC Subgraphs

Every other SCC has fake upstream vertices representing vertices from previous SCCs.

Structure:
- Auto-starting vertex (index 0) = fake starting vertex (not in original graph)
- Upstream vertices = fake starting vertices from previous SCCs
- Internal vertices = actual SCC members
- Downstream vertices = fake absorbing vertices for future SCCs

Vertex categories:
- upstream = [vertices from previous SCCs]
- upstream_connecting = [internal vertices receiving edges from upstream]
- internal = [actual SCC member vertices]
- downstream_connecting = [internal vertices sending edges to downstream]
- downstream = [vertices in future SCCs]