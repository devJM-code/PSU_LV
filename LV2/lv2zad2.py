import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),
delimiter=",", skiprows=1)

#print(data)
mpg = data[:,0:1]
hp = data[:,3:4]
wt = data[:,5:6]
cyl = data[:,1:2]
#print(mpg)
#print(hp)

#plt.scatter(mpg, hp,linewidths=wt)
plt.scatter(mpg, hp, c='purple', s=wt*12, cmap='plasma')
plt.show()

print(np.min(mpg))
print(np.max(mpg))
print(np.mean(mpg))

'''for a in data[:,0:1]:
if a == 6:
mpg1= a[:,0:1]
'''
i = -1
mpg2 = []
for a in cyl:
i += 1
if a == 6:
mpg2.append(mpg[i])

print(np.min(mpg2))
print(np.max(mpg2))
print(np.mean(mpg2))
