add_executable(example_basic basic_example.c gaussian_elimination_stub.c)
target_link_libraries(example_basic markov_cache)

install(TARGETS example_basic
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
