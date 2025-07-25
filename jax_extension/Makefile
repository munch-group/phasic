CXX = g++
CXXFLAGS = -O3 -fPIC -shared -std=c++17 -rdynamic -pthread -I./ -I../.pixi/envs/default/include -I../.pixi/envs/default/include/boost -L../.pixi/envs/default/lib -Wall
LIBS = -lhdf5_cpp -lhdf5 -lz -lssl -lcrypto

# Original PMF function
jax_graph_method_pmf.so: jax_graph_method_pmf.cpp
	$(CXX) $(CXXFLAGS) jax_graph_method_pmf.cpp -o jax_graph_method_pmf.so $(LIBS)

# Separated architecture components
user_graph_api.o: user_graph_api.cpp user_graph_api.h
	$(CXX) -O3 -fPIC -std=c++17 -I./ -I../.pixi/envs/default/include -Wall -c user_graph_api.cpp -o user_graph_api.o

separated_graph_pmf.so: separated_graph_pmf.cpp user_graph_api.o user_graph_api.h
	$(CXX) $(CXXFLAGS) separated_graph_pmf.cpp user_graph_api.o -o separated_graph_pmf.so $(LIBS)

# Build all components
all: jax_graph_method_pmf.so separated_graph_pmf.so

# Test the separated system
test: separated_graph_pmf.so
	python3 example_coalescent.py

clean:
	rm -f *.so *.o model.bin.z /tmp/user_graph_*.so

.PHONY: all test clean