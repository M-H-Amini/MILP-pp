#!/bin/bash

docker build -t mh_milp .

docker run --gpus device=2 -v ${PWD}:/usr/src/app --name mh_milp_pp_2  --shm-size=64g -it --rm mh_milp bash -l