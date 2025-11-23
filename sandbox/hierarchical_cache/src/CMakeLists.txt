set(SOURCES
    cache.c
    graph.c
    csr.c
    serialization.c
    compression.c
    varint.c
    lru.c
    scc.c
    hash.c
    utils.c
)

add_library(markov_cache ${SOURCES})

target_include_directories(markov_cache
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(markov_cache
    PUBLIC
        PkgConfig::ZSTD
        PkgConfig::SQLITE3
        OpenSSL::Crypto
    PRIVATE
        m
)

set_target_properties(markov_cache PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/markov_cache/markov_cache.h"
)

install(TARGETS markov_cache
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/markov_cache
)
