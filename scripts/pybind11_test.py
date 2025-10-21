
import phasic

print(phasic.__file__)

graph = phasic.Graph(3)
print(graph)

print(dir(graph))

print(graph.vertices_length())

vertex = graph.create_vertex([0, 0, 0])
print(vertex)

print(graph.vertices_length())

exp = graph.expected_waiting_time([])
print(exp)

exp = graph.expected_waiting_time([])
print(exp)

