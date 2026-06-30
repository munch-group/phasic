#!/usr/bin/env bash

cd docs
#rm -f api/_styles-quartodoc.css api/_sidebar.yml #*.qmd
# Python API (quartodoc) + C/C++ API (doxygen XML -> gen_cpp_api.py), then render.
quartodoc build && quartodoc interlinks \
  && doxygen Doxyfile && python ../scripts/gen_cpp_api.py \
  && quarto render
cd ..
