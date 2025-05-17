#!/usr/bin/env bash

code="import toml ; print(toml.load('pyproject.toml')['project']['version'])"
tag="v$(python -c \"$code\")"    
echo $tag

if [[ `git status --porcelain` ]]; then
  echo "Changes to repo must be pushed first."
else
    # tag="v$(python setup.py --version)"
    command=(python -c "import toml ; print(toml.load('pyproject.toml')['project']['version'])")
    tag="v$(${command[@]})"    
    echo $tag
    # git tag -a $tag -m "${1:-Release}" && git push origin --tags
fi
# pycode="from importlib.metadata import version ; print(version())"
# version=python -c \"from importlib.metadata import version ; print(version($package))\")
# tag="v$(python -c \"from importlib.metadata import version ; print(version($package))\")"

# command=(python -c "import toml ; print(toml.load('pyproject.toml')['project']['version'])")
# tag="v$(${command[@]})"


