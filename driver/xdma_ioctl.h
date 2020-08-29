#ifndef XDMA_IOCTL_H
#define XDMA_IOCTL_H
#include <linux/ioctl.h>

struct xdma_huge {
unsigned long addr;
unsigned long size;
};

struct xdma_huge_mapping {
unsigned long npages;
unsigned long* dma_addr;
};

#define IOCTL_XDMA_BUFFER_SET  _IOW('q', 1, struct xdma_huge*)
#define IOCTL_XDMA_MAPPING_GET _IOR('q', 2, struct xdma_huge_mapping*)
#define IOCTL_XDMA_RELEASE     _IO ('q', 3)


#endif
