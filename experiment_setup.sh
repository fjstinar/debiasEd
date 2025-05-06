#!/bin/bash
(cd "../" && git pull https://Jadouille:"$1"@github.com/epfl-ml4ed/redacted-groupies.git)
echo new arguments
echo "$2" "$3" "$4" "$5" "$6"
pip3 install -r ../requirements.txt

python script_classification.py --cluster "$2" "$3" "$4" "$5" "$6"

