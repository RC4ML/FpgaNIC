def amax6(cycles,length):
	speed = 1.0*length/1024/(cycles/1.77/1e9)
	print(cycles,"amax6 speed:",speed)

def amax4(cycles,length):
	speed = 1.0*length/1024/(cycles/1.41/1e9)
	print(cycles,"amax4 speed:",speed)

def fpga(cycles,length):
	speed = 1.0*length/1024/(cycles*4/1e9)
	print(cycles,"fpga speed:",speed)

def ms_amax6(a,b):
	t=(b-a)/1.77/1e6
	print("ms",abs(t))
def ms_amax4(a,b):
	t=(b-a)/1.41/1e6
	print("ms",abs(t))



fpga(0x416bbc3,2048)


# ms_amax6(18496590653,18497381872)
# ms_amax4(14063264046,14084028421)
# ms_amax6(18422549016,18421729120)
