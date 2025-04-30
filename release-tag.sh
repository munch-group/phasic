#!/usr/bin/env bash

if [[ `git status --porcelain` ]]; then
  echo "Changes to repo must be pushed first."
else
    tag="v$(python setup.py --version)"
    git tag -a $tag -m "${1:-Release}" && git push origin --tags
fi