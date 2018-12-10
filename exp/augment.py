
f = open('note_random.txt','r')
for line in f:
	token = line.strip().split(',')
	print(token)
	
	time = token[0]
	act = token[1:4]
	obs = token[4:]
	
	print(act)