CXX = g++
# CXXFLAGS = -std=c++17 -I./ -I./cereal -Wall
CXXFLAGS = -O3 -fPIC -shared -std=c++17 -rdynamic -pthread -I./ -I../.pixi/envs/default/include -I../.pixi/envs/default/include/boost -L../.pixi/envs/default/lib -Wall
LIBS = -lhdf5_cpp -lhdf5 -lz -lssl -lcrypto

main: jax_graph_method_pmf.cpp
	$(CXX) $(CXXFLAGS) jax_graph_method_pmf.cpp -o jax_graph_method_pmf.so $(LIBS)

clean:
	rm -f jax_graph_method_pmf model.bin.z
