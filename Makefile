bootstrap :
	python nodestart.py -pp 1000
nodeoneoftwo :
	python nodestart.py -i 192.168.1.82  -p 1000 -pp 2000 -np 1 -pn 0
nodetwooftwo :
	python nodestart.py -i 192.168.1.82  -p 1000 -pp 3000 -np 1 -pn 1
bootstrapprune :
	python nodestartwithpruning.py -pp 1000
nodeoneoftwoprune :
	python nodestartwithpruning.py -i 192.168.1.82  -p 1000 -pp 2000 -np 1 -pn 0
nodetwooftwoprune :
	python nodestartwithpruning.py -i 192.168.1.82  -p 1000 -pp 3000 -np 1 -pn 1
