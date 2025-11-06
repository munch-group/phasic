"""Debug graph clone - check vertex counts"""

from phasic import Graph

# Create a simple graph
g = Graph(state_length=1)
print(f"After Graph(state_length=1): vertices_length = {g.vertices_length()}")

v0 = g.starting_vertex()
print(f"After getting starting_vertex: vertices_length = {g.vertices_length()}")

v1 = g.find_or_create_vertex([1])
print(f"After find_or_create_vertex([1]): vertices_length = {g.vertices_length()}")

v0.add_edge(v1, 1.0)
print(f"After add_edge: vertices_length = {g.vertices_length()}")

print(f"\nOriginal graph: {g.vertices_length()} vertices")

# List all vertices
print("Vertices in original:")
for i, v in enumerate(g.vertices()):
    print(f"  {i}: state={v.state()}")

# Now clone
print("\nCloning...")
g2 = g.clone()
print(f"Clone: {g2.vertices_length()} vertices")

# List all vertices in clone
print("Vertices in clone:")
for i, v in enumerate(g2.vertices()):
    print(f"  {i}: state={v.state()}")

# Check starting vertex
print(f"\nOriginal starting vertex state: {g.starting_vertex().state()}")
print(f"Clone starting vertex state: {g2.starting_vertex().state()}")

# Try adding a vertex to original only
print("\nAdding vertex [2] to original...")
v2 = g.find_or_create_vertex([2])
print(f"Original: {g.vertices_length()} vertices")
print(f"Clone: {g2.vertices_length()} vertices")

if g2.vertices_length() == 1:
    print("\n✅ Clone is independent!")
else:
    print(f"\n❌ Clone has {g2.vertices_length()} vertices, should have 1!")
