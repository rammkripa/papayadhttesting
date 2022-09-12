bootstrap :
	python nodestart.py -pp 1000
nodeoneoftwo :
	python nodestart.py -i 10.105.165.155  -p 1000 -pp 2000 -np 1 -pn 0
nodetwooftwo :
	python nodestart.py -i 10.105.165.155  -p 1000 -pp 3000 -np 1 -pn 1
twonodes :
	make nodeoneoftwo
	make nodetwooftwo
