#!/bin/bash

rm output.txt

for  i in `seq 12`; do
	echo -e '\n\nProblem '$i >> output.txt;
	./pymplex.py data/problem_$i.json >> output.txt;
done

for  i in `seq 4`; do
	echo -e '\n\nProblem '$i >> output.txt;
	./pymplex.py data/problem_0$i.json >> output.txt;
done
