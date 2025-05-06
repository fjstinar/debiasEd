#!/bin/bash
git add *
git commit -m "experiment push on cluster"
git push origin main
python cluster/runai_arguments.py --name maigroupies-"$1" --args "$2"
kubectl create -f cluster/runai_config.yaml

# Arguments
# 1. name of the image after ic-registry.epfl.ch/d-vet/
# 2. name of the runai container 
# 3. List of arguments in one big string:
    # 1. github ssh key
    # 2. --baseline or --preprocessing

