  The encoding: a canonical parent vector

  Hold the topology as an ancestor/parent vector — the fixed‑length, canonical,
  integer representation that also makes a recombination a local edit:

  - Nodes: leaves 0..n−1, internal n..2n−2; state[i] = parent(i), sentinel for
  not‑yet‑coalesced roots. Length 2n−1.
  - Canonicalize the internal numbering deterministically (by coalescence rank if
  your rates are time/rank‑dependent — the usual coalescent case — else by
  min‑leaf‑in‑clade), so equivalent trees collapse to one vector. This is the
  find_or_create_vertex identity constraint from last turn; skip it and you get
  the n!‑blowup.
  - A recombination detaches a subtree and re‑attaches it = rewrite a handful of
  parent entries. Topology changed ⇔ canonicalized vector changed ⇒ route to the
  absorbing vertex; unchanged ⇒ a transient transition. Same edge‑table idea as
  the tskit tree sequence from two turns ago, collapsed to a single marginal tree.

  State count = labelled topologies you track: unranked (2n−3)!! = 15, 105, 945,
  10395 for n = 4,5,6,7; ranked n!(n−1)!/2^(n−1) = 18, 180, … is larger. Both are
  fine at the small nr_samples your two‑locus models use (recollection). The
  sequence/type block (last turn's Property/StateIndexer, grounded in
  state_indexing.py) gets appended only if sequence type changes the rates;
  neutral marks stay off the state and are read off the path.