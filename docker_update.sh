#!/bin/bash
docker build --no-cache --tag mai-groupies -f Dockerfile . --build-arg LDAP_GROUPNAME=ml4ed \
    --build-arg LDAP_GID=11159 \
    --build-arg LDAP_USERNAME=cock \
    --build-arg LDAP_UID=195749
# docker tag mai-conf ic-registry.epfl.ch/d-vet/mai-groupies-"$1"
# docker push ic-registry.epfl.ch/d-vet/mai-groupies-"$1"
docker tag mai-groupies registry.rcp.epfl.ch/ml4ed/mai-groupies
docker push registry.rcp.epfl.ch/ml4ed/mai-groupies

# I use the argument to build one image per dataset, such as to not use too much memory

