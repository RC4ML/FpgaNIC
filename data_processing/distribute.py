import numpy as np
f = open("./data_processing/latency.txt")

data = []
line = f.readline()
while line:
	if line!="\n":
		data.append(float(line))
	line = f.readline()
f.close()

num = len(data)
print("number of data",num)
data.sort()
# for it in data:
# 	print(it)

index_1 = int(1.0/100*num)
index_5 = int(5.0/100*num)
index_25 = int(25.0/100*num)
index_75 = int(75.0/100*num)
index_95 = int(95.0/100*num)
index_99 = int(99.0/100*num)
index_mid = int(0.5*num)

print(index_1)
print(index_5)
print(index_25)
print(index_mid)
print(index_75)
print(index_95)
print(index_99)

print(data[index_1],end=" ")
print(data[index_5],end=" ")
print(data[index_25],end=" ")
print(data[index_mid],end=" ")
print(data[index_75],end=" ")
print(data[index_95],end=" ")
print(data[index_99],end=" ")

# boundary_low = 1
# boundary_high = 100-boundary_low
# per = boundary_low/100
# index_low = int(per*num)
# index_high = int((1-per)*num)
# print("low index:",index_low," high index:",index_high)

# print("avg:"+str(np.mean(data)))
# print("low"+str(boundary_low)+":"+str(data[index_low]))
# print("high"+str(boundary_high)+":"+str(data[index_high]))
# print("min:"+str(data[0]))
# print("max:"+str(data[num-1]))