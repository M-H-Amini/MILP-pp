#!/bin/bash

docker build -t mh_milp .

docker run --gpus device=2 -v ${PWD}:/usr/src/app  --shm-size=64g -it --rm --name mh_milp_container mh_milp bash -l