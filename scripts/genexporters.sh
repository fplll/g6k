#/usr/bin/bash

cat stats_list.txt | while read statname
do
sed s/REPLACEME/$statname/g stats_exp.txt
done
