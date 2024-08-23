#!/usr/bin/env bash
docker run --rm -v $HOME/.ssh:/tmp/.ssh -v $(pwd):$(pwd) -w $(pwd) -i -t -p 8888:8888 kaspermunch/phasetype

