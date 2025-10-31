# C-Level Debugging Guide for Vertex Bypass Bug

This guide shows exactly where to add printf debugging statements in the C code to trace the bug.

---

## Location: `src/c/phasic.c`

Function: `_ptd_graph_reward_transform()`
Lines: ~2151-2312 (with my attempted fixes) or ~2151-2294 (original)

---

## Debugging Statements to Add

### 1. At the start of the main bypass loop (around line 2160)

```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0 || bypassed[i]) {
        continue;
    }

    // ADD THIS:
    printf("\n=== BYPASSING VERTEX %zu ===\n", i);
    printf("  State: [%d]\n", vertices[i]->state[0]);
    printf("  Reward: %.6f\n", rewards[i]);
    printf("  Parents: %zu\n", vertex_parents_length[i]);
    printf("  Children: %zu\n", vertex_edges_length[i]);
```

### 2. At the start of the parent loop (around line 2174)

```c
for (size_t p = 0; p < my_parents_length; ++p) {
    struct arr_p me_to_parent = vertex_parents[i][p];
    struct ptd_vertex *parent_vertex = me_to_parent.p;
    size_t parent_vertex_index = parent_vertex->index;

    // ADD THIS:
    printf("\n  Parent %zu: vertex %zu\n", p, parent_vertex_index);
    printf("    State: [%d]\n", parent_vertex->state[0]);
    printf("    Bypassed: %s\n", bypassed[parent_vertex_index] ? "YES" : "NO");
    printf("    Reward: %.6f\n", rewards[parent_vertex_index]);

    if (bypassed[parent_vertex_index]) {
        // ADD THIS:
        printf("    ⚠️  SKIPPING bypassed parent\n");
        continue;
    }

    printf("    Processing edges from parent %zu to vertex %zu's children\n",
           parent_vertex_index, i);
```

### 3. When creating/updating edges (around line 2237-2255)

In the merge case (when parent already has edge to same child):

```c
if (me_to_child_v == parent_to_child_v) {
    // ADD THIS:
    printf("    MERGE: parent %zu → child %zu (already connected)\n",
           parent_vertex_index, me_to_child_v->index);
    printf("      Old prob: %.6f\n", parent_to_child.prob);
    printf("      Add prob: %.6f * %.6f = %.6f\n",
           me_to_child_p, parent_weight_to_me, me_to_child_p * parent_weight_to_me);

    new_parent_children[vertex_edges_length[parent_vertex_index]].to = parent_to_child_v;
    new_parent_children[vertex_edges_length[parent_vertex_index]].prob =
            parent_to_child.prob + me_to_child_p * parent_weight_to_me;

    // ADD THIS:
    printf("      New prob: %.6f\n",
           new_parent_children[vertex_edges_length[parent_vertex_index]].prob);
```

In the new edge case (when parent doesn't have edge to child):

```c
} else if (me_to_child_v < parent_to_child_v) {
    // ADD THIS:
    printf("    NEW EDGE: parent %zu → child %zu\n",
           parent_vertex_index, me_to_child_v->index);
    printf("      Prob: %.6f * %.6f = %.6f\n",
           me_to_child_p, parent_weight_to_me, me_to_child_p * parent_weight_to_me);
```

### 4. At normalization (around line 2284-2286)

```c
// Make sure parent has rate of 1
printf("  Normalizing parent %zu edges: total_prob = %.6f\n",
       parent_vertex_index, new_parent_total_prob);

for (size_t j = 0; j < vertex_edges_length[parent_vertex_index]; ++j) {
    double old_prob = new_parent_children[j].prob;
    new_parent_children[j].prob /= new_parent_total_prob;
    // ADD THIS:
    printf("    Edge %zu: %.6f → %.6f (÷ %.6f)\n",
           j, old_prob, new_parent_children[j].prob, new_parent_total_prob);
}
```

### 5. After marking vertex as bypassed (around line 2308)

```c
// Mark this vertex as bypassed
bypassed[i] = true;

// ADD THIS:
printf("\n✓ Vertex %zu bypassed and marked\n", i);
printf("  Bypassed array: [");
for (size_t k = 0; k < vertices_length; ++k) {
    printf("%d%s", bypassed[k] ? 1 : 0, k < vertices_length-1 ? ", " : "");
}
printf("]\n");

// Break from for loop to restart
break;
```

### 6. Final graph construction (around line 2325-2350)

```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (vertices[i] == graph->starting_vertex) {
        continue;
    }

    if (bypassed[i]) {
        // ADD THIS:
        printf("FINAL: Skipping bypassed vertex %zu\n", i);
        continue;
    }

    // ADD THIS:
    printf("FINAL: Adding vertex %zu (state=[%d]) with reward=%.6f\n",
           i, vertices[i]->state[0], rewards[i]);
```

And in the edge creation loop:

```c
for (size_t j = 1; j < vertex_edges_length[i] - 1; ++j) {
    // ADD THIS:
    printf("FINAL: Adding edge %zu → %zu, prob=%.6f/%.6f = %.6f\n",
           i, vertex_edges[i][j].to->index,
           vertex_edges[i][j].prob, rewards[i],
           vertex_edges[i][j].prob / rewards[i]);

    ptd_graph_add_edge(...);
}
```

---

## How to Run with Debugging

1. **Add the printf statements** to `src/c/phasic.c`

2. **Rebuild**:
   ```bash
   pip install -e . --no-build-isolation
   ```

3. **Run the test**:
   ```bash
   python /tmp/diagnose_vertex_bypass.py 2>&1 | tee /tmp/bypass_debug.log
   ```

4. **Examine the output**:
   ```bash
   less /tmp/bypass_debug.log
   ```

---

## What to Look For

### Expected Pattern (Working Case - 2 bypasses):

```
=== BYPASSING VERTEX 1 ===
  Parent 0: vertex 0, Bypassed: NO
    Processing edges...
    NEW EDGE: parent 0 → child 2
✓ Vertex 1 bypassed

=== BYPASSING VERTEX 2 ===
  Parent 0: vertex 0, Bypassed: NO  ← Parent list was updated!
    Processing edges...
    NEW EDGE: parent 0 → child 3
✓ Vertex 2 bypassed
```

### Broken Pattern (3 bypasses):

```
=== BYPASSING VERTEX 1 ===
  Parent 0: vertex 0, Bypassed: NO
    Processing edges...
    NEW EDGE: parent 0 → child 2
✓ Vertex 1 bypassed

=== BYPASSING VERTEX 2 ===
  Parent 0: vertex 1, Bypassed: YES  ← STALE! Should be vertex 0
    ⚠️  SKIPPING bypassed parent
  (No edges created - vertex 2 becomes disconnected!)
✓ Vertex 2 bypassed

=== BYPASSING VERTEX 3 ===
  Parent 0: vertex 2, Bypassed: YES  ← STALE! Should be vertex 0
    ⚠️  SKIPPING bypassed parent
  (No edges created - vertex 3 becomes disconnected!)
✓ Vertex 3 bypassed

FINAL: No edges from vertex 0 to vertex 4!
Result: Disconnected graph
```

---

## Key Insights from Debug Output

You should see:

1. **Stale parent references**: `vertex_parents[2]` shows vertex 1, even after vertex 1 was bypassed

2. **Skipped edge creation**: When all parents are bypassed, NO edges get created for that vertex

3. **Missing final edge**: The final graph has vertices 0 and 4, but no edge connecting them!

4. **Parent list never updated**: The `vertex_parents[]` array is built once at lines 2135-2146 and never updated during bypass operations

---

## The Fix Needed

After seeing the debug output, the fix becomes clearer:

**Option A**: Don't skip bypassed parents - instead, recursively resolve them

```c
struct ptd_vertex *get_actual_parent(size_t i, size_t p_idx) {
    struct ptd_vertex *parent = vertex_parents[i][p_idx].p;
    size_t parent_idx = parent->index;

    // Recursively resolve bypassed parents
    while (bypassed[parent_idx]) {
        // Get parent's first parent (assumes linear chain)
        parent = vertex_parents[parent_idx][0].p;
        parent_idx = parent->index;
    }

    return parent;
}
```

**Option B**: Rebuild parent lists after each bypass (expensive but correct)

```c
// After bypassing vertex i and before breaking:
for (size_t j = 0; j < vertices_length; ++j) {
    vertex_parents_length[j] = 0;
}

// Rebuild from edges
for (size_t j = 0; j < vertices_length; ++j) {
    for (size_t e = 1; e < vertex_edges_length[j] - 1; ++e) {
        size_t child_idx = vertex_edges[j][e].to->index;
        vertex_parents[child_idx][vertex_parents_length[child_idx]].p = vertices[j];
        vertex_parents[child_idx][vertex_parents_length[child_idx]].arr_c_index = e;
        vertex_edges[j][e].arr_p_index = vertex_parents_length[child_idx];
        vertex_parents_length[child_idx]++;
    }
}
```

---

## Test After Fix

Run the diagnostic script again to verify:
- PDF integral ≈ 1.0
- Sample mean > 0
- All test cases pass

---

## Additional Test Cases

Test with different patterns:

```python
# Non-consecutive bypasses (should work)
rewards = [1, 0, 1, 0, 1]  # Should work - not consecutive

# 4 consecutive bypasses
rewards = [1, 0, 0, 0, 0, 1]  # Should fail with current code

# All bypasses except endpoints
rewards = [1, 0, 0, 0, 0, 0, 1]  # Should fail with current code
```
