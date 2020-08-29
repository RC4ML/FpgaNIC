# dma-driver

## Build Kernel Module/Driver

1. Install prerequisites, e.g. on Ubuntu install the following packages:
```
$ apt install build-essential linux-headers-generic
```

2. Build kernel module
```
$ cd dma/driver/driver
$ make
```


## Setup Hugepages
1. Create a group for users of hugepages, and retrieve its GID (in this example, 1001) then add yourself to the group.
```
$ groupadd hugetlbfs

$ getent group hugetlbfs

$ adduser dasidler hugetlbfs
```

2. Edit `/etc/sysctl.conf` and add this text to specify the number of pages you want to reserve (see page-size)
```
# Allocate 8192*2MiB for HugePageTables
vm.nr_hugepages = 8192

# Members of group hugetlbfs(1001) can allocate "huge" shared memory segments
vm.hugetlb_shm_group = 1001
```
3. Create a mount point for the file system
```
$ mkdir /media/huge
```

4. Add this line in `/etc/fstab` (The mode 1770 allows anyone in the group to create files but not unlink or rename each other's files.)
```
# hugetlbfs
hugetlbfs /media/huge hugetlbfs mode=1770,gid=1001 0 0
```

5. Reboot

6. Add the following line to `/etc/security/limits.conf` to configure the amount of memory a user can lock, so an application can't crash your operating system by locking all the memory. 
```
@hugetlbfs	hard	memlock		1048576
```

## Build Example Application
1. Install prerequisites, e.g. on Ubuntu install the following packages:
```
$ apt install libboost-program-options-dev cmake
```
2. Compile example application
```
$ cd dma-driver/sw
$ mkdir build && cd build
$ cmake ../src
$ make
```

## Run Example Application/Benchmark
1. Load kernel module if not loaded yet.
```
$ cd dma-driver/driver
$ insmod xdma_driver.ko
```
2. Run the Application (requires root permission)
```
$ cd dma-driver/sw/build
$ ./dma-example
```

## Read FPGA Debug Registers
1. Load kernel module if not loaded yet.
```
$ cd dma-driver/driver
$ insmod xdma_driver.ko
```
2. Run the Application (requires root permission)
```
$ cd dma-driver/sw/build
$ ./debug
```
