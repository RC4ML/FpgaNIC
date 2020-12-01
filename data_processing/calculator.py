def amax6(cycles,length):
	speed = 1.0*length/1024/(cycles/1.77/1e9)
	print(cycles,"amax6 speed:",speed)

def amax4(cycles,length):
	speed = 1.0*length/1024/(cycles/1.41/1e9)
	print(cycles,"amax4 speed:",speed)

def fpga(cycles,length):
	speed = 1.0*length/1024/(cycles*4/1e9)
	print(cycles,"fpga speed:",speed)




amax4(69248475,500)

