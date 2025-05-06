# cd ./cock/redacted-groupies
echo
echo cloning
(cd "/code/" && git clone https://Jadouille:"$1"@github.com/epfl-ml4ed/redacted-groupies.git)
for i in `ls`; do echo $i; done;
echo 

echo entering code
cd /code/redacted-groupies/src
bash ../experiment_setup.sh "$1" "$2" "$3" "$4" "$5" "$6"

