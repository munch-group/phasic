#!/usr/bin/env bash


gh release create --latest "v$(python setup.py --version | tail -n 1)" --title "v$(python setup.py --version | tail -n 1)" --notes ""
