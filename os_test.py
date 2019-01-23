import os

dataPaths = os.listdir('/home/mookse/workspace/MinKNOW/selected/')

dataPaths = [os.path.join('/home/mookse/workspace/MinKNOW/selected/', dataPath) for dataPath in dataPaths]

files = [os.listdir(dataPath) for dataPath in dataPaths]

files_full = []
 
dataPaths = [ [dataPaths[i]]*len(files[i]) for i in range(len(files)) ]

#print([os.path.join(dataPaths[i], files[i]) for i in range(len(files))])
print(zip(dataPaths, files))

for sublist in zip(dataPaths, files):
	for i in range(len(sublist[0])):
		files_full.append(os.path.join(sublist[0][i], sublist[1][i]))

print(files_full)
