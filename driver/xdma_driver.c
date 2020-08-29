#include <asm/io.h>
#include <linux/clk.h>
#include <linux/device.h>
#include <linux/hrtimer.h>
#include <linux/init.h>     /* Needed for the macros */
#include <linux/interrupt.h>
#include <linux/ioport.h>
#include <linux/kernel.h>   /* Needed for KERN_INFO */
#include <linux/miscdevice.h>
#include <linux/module.h>   /* Needed by all modules */
#include <linux/pci.h>
#include <linux/scatterlist.h>
#include <linux/sched.h>
#include <linux/types.h>
#include <linux/cdev.h>
#include <linux/pagemap.h>
#include <linux/vmalloc.h>
#include "xdma_ioctl.h"
#include <linux/mm.h>
#include <linux/version.h>

#include <asm/io.h>

#define PER_FRAME_DEBUG 0

#define MASK(n) (BIT(n) - 1)

/* 10ms (in ns) */
#define POLL_PERIOD (10 * 1000 * 1000)

#define DRV_NAME "xdma_driver"
#define XDMA_CHANNEL_NUM_MAX (4)
#define MAX_NUM_ENGINES (XDMA_CHANNEL_NUM_MAX * 2)
#define C2H_CHANNEL_OFFSET 0x0000 //TODO is this not the other way around??
#define H2C_CHANNEL_OFFSET 0x1000
#define SGDMA_OFFSET_FROM_CHANNEL 0x4000
#define CHANNEL_SPACING 0x100
#define TARGET_SPACING 0x1000
#define MAX_USER_IRQ 16
#define XDMA_BAR_NUM (6)
#define XDMA_BAR_SIZE (0x8000UL)
#define IRQ_BLOCK_ID 0x1fc20000L
#define CONFIG_BLOCK_ID 0x1fc30000UL

/* upper 16-bits of engine identifier register */
#define XDMA_ID_H2C 0x1fc0U
#define XDMA_ID_C2H 0x1fc1U

/* Target internal components on XDMA control BAR */
#define XDMA_OFS_INT_CTRL   (0x2000UL)
#define XDMA_OFS_CONFIG     (0x3000UL)

/* bits of the SGDMA descriptor control field */
#define XDMA_DESC_MAGIC     (0xAD4B0000UL)
#define XDMA_DESC_STOPPED   (1UL << 0)
#define XDMA_DESC_COMPLETED (1UL << 1)
#define XDMA_DESC_EOP       (1UL << 4)

/* bits of the SG DMA control register */
#define XDMA_CTRL_RUN_STOP          (1UL << 0)
#define XDMA_CTRL_IE_DESC_STOPPED       (1UL << 1)
#define XDMA_CTRL_IE_DESC_COMPLETED     (1UL << 2)
#define XDMA_CTRL_IE_DESC_ALIGN_MISMATCH    (1UL << 3)
#define XDMA_CTRL_IE_MAGIC_STOPPED      (1UL << 4)
#define XDMA_CTRL_IE_IDLE_STOPPED       (1UL << 6)
#define XDMA_CTRL_IE_READ_ERROR         (0x1FUL << 9)
#define XDMA_CTRL_IE_DESC_ERROR         (0x1FUL << 19)
#define XDMA_CTRL_NON_INCR_ADDR         (1UL << 25)
#define XDMA_CTRL_POLL_MODE_WB          (1UL << 26)

/* bits of the SG DMA status register */
#define XDMA_STAT_BUSY          (1UL << 0)
#define XDMA_STAT_DESC_STOPPED      (1UL << 1)
#define XDMA_STAT_DESC_COMPLETED    (1UL << 2)
#define XDMA_STAT_ALIGN_MISMATCH    (1UL << 3)
#define XDMA_STAT_MAGIC_STOPPED     (1UL << 4)
#define XDMA_STAT_FETCH_STOPPED     (1UL << 5)
#define XDMA_STAT_IDLE_STOPPED      (1UL << 6)
#define XDMA_STAT_READ_ERROR        (0x1FUL << 9)
#define XDMA_STAT_DESC_ERROR        (0x1FUL << 19)

/* obtain the 32 most significant (high) bits of a 32-bit or 64-bit address */
#define PCI_DMA_H(addr) ((addr >> 16) >> 16)
/* obtain the 32 least significant (low) bits of a 32-bit or 64-bit address */
#define PCI_DMA_L(addr) (addr & 0xffffffffUL)

static long char_xdma_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
int xdma_open(struct inode *inode, struct file *file);
int xdma_release(struct inode *inode, struct file *file);

/**
 * Descriptor for a single contiguous memory block transfer.
 *
 * Multiple descriptors are linked by means of the next pointer. An additional
 * extra adjacent number gives the amount of extra contiguous descriptors.
 *
 * The descriptors are in root complex memory, and the bytes in the 32-bit
 * words must be in little-endian byte ordering.
 */
struct xdma_desc {
    u32 control;
    u32 bytes;          /* transfer length in bytes */
    u32 src_addr_lo;    /* source address (low 32-bit) */
    u32 src_addr_hi;    /* source address (high 32-bit) */
    u32 dst_addr_lo;    /* destination address (low 32-bit) */
    u32 dst_addr_hi;    /* destination address (high 32-bit) */
    /*
     * next descriptor in the single-linked list of descriptors;
     * this is the PCIe (bus) address of the next descriptor in the
     * root complex memory
     */
    u32 next_lo;        /* next desc address (low 32-bit) */
    u32 next_hi;        /* next desc address (high 32-bit) */
} __packed;

#define DESC_PER_PAGE (PAGE_SIZE / sizeof(struct xdma_desc))

/* 32 bytes (four 32-bit words) or 64 bytes (eight 32-bit words) */
struct xdma_result {
    u32 status;
    u32 length;
    u32 reserved_1[6];  /* padding */
} __packed;

#define RESULTS_PER_PAGE (PAGE_SIZE / sizeof(struct xdma_result))

/*
 * SG DMA Controller status and control registers
 *
 * These registers make the control interface for DMA transfers.
 *
 * It sits in End Point (FPGA) memory BAR[0] for 32-bit or BAR[0:1] for 64-bit.
 * It references the first descriptor which exists in Root Complex (PC) memory.
 *
 * @note The registers must be accessed using 32-bit (PCI DWORD) read/writes,
 * and their values are in little-endian byte ordering.
 */
struct engine_regs {
    u32 identifier;
    u32 control;
    u32 control_w1s;
    u32 control_w1c;
    u32 reserved_1[12]; /* padding */

    u32 status;
    u32 status_rc;
    u32 completed_desc_count;
    u32 alignments;
    u32 reserved_2[14]; /* padding */

    u32 poll_mode_wb_lo;
    u32 poll_mode_wb_hi;
    u32 interrupt_enable_mask;
    u32 interrupt_enable_mask_w1s;
    u32 interrupt_enable_mask_w1c;
    u32 reserved_3[9];  /* padding */

    u32 perf_ctrl;
    u32 perf_cyc_lo;
    u32 perf_cyc_hi;
    u32 perf_dat_lo;
    u32 perf_dat_hi;
    u32 perf_pnd_lo;
    u32 perf_pnd_hi;
} __packed;

struct engine_sgdma_regs {
    u32 identifier;
    u32 reserved_1[31]; /* padding */

    /* bus address to first descriptor in Root Complex Memory */
    u32 first_desc_lo;
    u32 first_desc_hi;
    /* number of adjacent descriptors at first_desc */
    u32 first_desc_adjacent;
    u32 credits;
} __packed;

/* This is the driver-private data for a device instance. */
struct dev_inst {
    unsigned long magic;        /* structure ID for sanity checks */
    struct pci_dev *pci_dev;    /* pci device struct from probe() */
    int instance;               /* instance number */
    struct cdev cdev;
    dev_t cdevno_base;          /* character device major:minor base */

    /* PCIe BAR management */
    void *__iomem bar[XDMA_BAR_NUM]; /* addresses for mapped BARs */
    int user_bar_idx;                /* BAR index of user logic */
    int config_bar_idx;              /* BAR index of XDMA config logic */
    int bypass_bar_idx;              /* BAR index of XDMA bypass logic */
    int regions_in_use;              /* flag if dev was in use during probe() */
    int got_regions;                 /* flag if probe() obtained the regions */

    /* Interrupt management */
    int irq_count;                /* interrupt counter */
    int irq_line;                 /* flag if irq allocated successfully */
    int msi_enabled;              /* flag if msi was enabled for the device */
    int msix_enabled;             /* flag if msi-x was enabled for the device */
    struct msix_entry entry[32];  /* msi-x vector/entry table */

    /* XDMA engine management */
    int h2c_count;
    int c2h_count;
    int h2c_engine;
    int c2h_engine;
    //int trace_engine;
    //uint32_t engine_irq_mask;
    struct engine_regs *h2c_regs;
    struct engine_regs *c2h_regs;
    //struct engine_sgdma_regs *sgdma_regs;

    /* DMA memory. */
    unsigned long huge_user_addr;
    unsigned long huge_size;
    unsigned long huge_npages;
    unsigned long huge_hnpages;
    struct page **huge_pages; /* theser are $KB pages use in 2MB pages */

    bool desc_bypass_enabled;
    int open_count;
};

/* The /dev/trace misc device node. */
struct miscdevice misc_dev_node;
struct miscdevice bypass_node;

/*****************************/
/* Utility functions         */
/*****************************/

static inline u32
build_u32(u32 hi, u32 lo) {
    return ((hi & 0xFFFFUL) << 16) | (lo & 0xFFFFUL);
}

static inline u64
build_u64(u64 hi, u64 lo) {
    return ((hi & 0xFFFFFFFULL) << 32) | (lo & 0xFFFFFFFFULL);
}

static inline int
read_bit(void *base, uintptr_t reg, int bit) {
    return (readl(base + reg) & BIT(bit)) != 0;
}

/**************************/
/* The /dev/xdma_control device. */
/**************************/
/* maps the PCIe BAR into user space for memory-like access using mmap() */
static int xdma_mmap(struct file *file, struct vm_area_struct *vma)
{
  int rc;
  //struct xdma_dev *lro;
  //struct xdma_char *lro_char = (struct xdma_char *)file->private_data;
  struct device *parent;
  struct dev_inst *inst; //= (struct dev_inst *)file->private_data;
  resource_size_t phys, psize;
  unsigned long off;
  //unsigned long phys;
  unsigned long vsize;
  //unsigned long psize;

  printk(KERN_INFO "xdma_mmap(%p ,%p)\n", file, vma);
  printk(KERN_INFO "MMAP %016lx-%016lx\n", vma->vm_start, vma->vm_end);
  
  /* We set the parent to the trace device on creation. */
  parent = misc_dev_node.parent;
  printk(KERN_INFO "Parent device %p\n", parent);
  inst = dev_get_drvdata(parent);
  printk(KERN_INFO "Instance private data at %p\n", inst);

  BUG_ON(!inst);
  printk(KERN_INFO "dev_inst: %p\n", inst);
  //TODO check magic??

  off = vma->vm_pgoff << PAGE_SHIFT;
  /* BAR physical address */
  //phys = pci_resource_start(inst->pci_dev, lro_char->bar) + off;
  vsize = vma->vm_end - vma->vm_start;
  printk(KERN_INFO "requesting resources, user bar %i\n", inst->user_bar_idx);
  printk(KERN_INFO "pci_dev: %p\n", inst->pci_dev);
  /* complete resource */
  phys = pci_resource_start(inst->pci_dev, inst->user_bar_idx);
  psize =   pci_resource_len(inst->pci_dev, inst->user_bar_idx);
  printk(KERN_INFO "physical bar address %p, size %lu\n", phys, psize);

  if (vsize > psize)
    return -EINVAL;
  /*
   * pages must not be cached as this would result in cache line sized
   * accesses to the end point
   */
  vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
  /*
   * prevent touching the pages (byte access) for swap-in,
   * and prevent the pages from being swapped out
   */
  vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
  /* make MMIO accessible to user space */
  rc = io_remap_pfn_range(vma, vma->vm_start, phys >> PAGE_SHIFT,
      vsize, vma->vm_page_prot);
  printk(KERN_INFO "vma=0x%p, vma->vm_start=0x%lx, phys=0x%lx, size=%lu = %d\n",
    vma, vma->vm_start, phys >> PAGE_SHIFT, vsize, rc);

  if (rc)
    return -EAGAIN;
  return 0;
}

/**************************/
/* The /dev/xdma_bypass device. */
/**************************/
/* maps the PCIe BAR into user space for memory-like access using mmap() */
static int xdma_bypass_mmap(struct file *file, struct vm_area_struct *vma)
{
  int rc;
  //struct xdma_dev *lro;
  //struct xdma_char *lro_char = (struct xdma_char *)file->private_data;
  struct device *parent;
  struct dev_inst *inst; //= (struct dev_inst *)file->private_data;
  resource_size_t phys, psize;
  unsigned long off;
  //unsigned long phys;
  unsigned long vsize;
  //unsigned long psize;

  printk(KERN_INFO "xdma_mmap(%p ,%p)\n", file, vma);
  printk(KERN_INFO "MMAP %016lx-%016lx\n", vma->vm_start, vma->vm_end);
  
  /* We set the parent to the trace device on creation. */
  parent = bypass_node.parent;
  printk(KERN_INFO "Parent device %p\n", parent);
  inst = dev_get_drvdata(parent);
  printk(KERN_INFO "Instance private data at %p\n", inst);

  BUG_ON(!inst);
  printk(KERN_INFO "dev_inst: %p\n", inst);
  //TODO check magic??

  off = vma->vm_pgoff << PAGE_SHIFT;
  /* BAR physical address */
  //phys = pci_resource_start(inst->pci_dev, lro_char->bar) + off;
  vsize = vma->vm_end - vma->vm_start;
  printk(KERN_INFO "requesting resources, user bar %i\n", inst->bypass_bar_idx);
  printk(KERN_INFO "pci_dev: %p\n", inst->pci_dev);
  /* complete resource */
  phys = pci_resource_start(inst->pci_dev, inst->bypass_bar_idx);
  psize =   pci_resource_len(inst->pci_dev, inst->bypass_bar_idx);
  printk(KERN_INFO "physical bar address %p, size %lu\n", phys, psize);

  if (vsize > psize)
    return -EINVAL;
  /*
   * pages must not be cached as this would result in cache line sized
   * accesses to the end point
   */
  vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
  /*
   * prevent touching the pages (byte access) for swap-in,
   * and prevent the pages from being swapped out
   */
  vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
  /* make MMIO accessible to user space */
  rc = io_remap_pfn_range(vma, vma->vm_start, phys >> PAGE_SHIFT,
      vsize, vma->vm_page_prot);
  printk(KERN_INFO "vma=0x%p, vma->vm_start=0x%lx, phys=0x%lx, size=%lu = %d\n",
    vma, vma->vm_start, phys >> PAGE_SHIFT, vsize, rc);

  if (rc)
    return -EAGAIN;
  return 0;
}

static const struct file_operations xdma_fops = {
    .owner = THIS_MODULE,
    .open = xdma_open,
    .release = xdma_release,
    .unlocked_ioctl = char_xdma_ioctl,
    .mmap = xdma_mmap,
};

static const struct file_operations xdma_bypass_fops = {
    .owner = THIS_MODULE,
    .open = xdma_open,
    .release = xdma_release,
    .mmap = xdma_bypass_mmap,
};

/* IOCTL */

static int ioctl_do_buffer_set(struct dev_inst *inst, unsigned long arg)
{
  int rc;
  struct xdma_huge huge;

  printk(KERN_INFO "IOCTL_XDMA_BUFFER_SET\n");
  rc = copy_from_user(&huge, (struct xdma_huge*)arg, sizeof(struct xdma_huge));
  if (rc != 0) {
    return rc;
  }

  printk(KERN_INFO "huge addr %p, size %d\n", huge.addr, huge.size);
  unsigned long npages;
  unsigned long buffer_start = huge.addr;
  unsigned long bufsize = huge.size;

  npages = 1 + (bufsize - 1) / PAGE_SIZE;
  printk(KERN_INFO "req npages %d\n", npages);
  printk(KERN_INFO "dev_inst ptr: %p\n", inst);
  inst->huge_pages = vmalloc(npages * sizeof(struct page*));

  printk(KERN_INFO "get_user_pages\n");
  down_read(&current->mm->mmap_sem);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 9, 0)
  rc = get_user_pages(buffer_start, npages, 1, inst->huge_pages, NULL);
#else
  rc = get_user_pages(current, current->mm, buffer_start, npages,
                      1 /* Write enable */, 0 /* Force */, inst->huge_pages, NULL);
#endif
  up_read(&current->mm->mmap_sem);
  if (rc == 0) {
    return -1;
  }
  npages = rc;
  printk(KERN_INFO "recv npages %d\n", npages);
  printk(KERN_INFO "size of unsigned long: %i\n", sizeof(unsigned long));

  int i = 0;
  int j = 0;
  for (i; i < npages; i++) {
    SetPageReserved(inst->huge_pages[i]);
    unsigned long dma_addr = page_to_phys(inst->huge_pages[i]);
    if (i % 512 == 0) {
      //printk(KERN_INFO "huge dma_addr: %p\n", dma_addr);
      j++;
    }
  }
  inst->huge_user_addr = huge.addr;
  inst->huge_size = npages * 4096;
  inst->huge_npages = npages;
  inst->huge_hnpages = j;
  printk(KERN_INFO "huge npages: %lu\n", j);

  return 0;
}

static int ioctl_do_mapping_get(struct dev_inst *inst, unsigned long arg)
{
  int rc;
  struct xdma_huge_mapping map;
  unsigned long* dma_addr;

  printk(KERN_INFO "IOCTL_MAPPING_GET\n");
  rc = copy_from_user(&map, (struct xdma_huge_mapping*)arg, sizeof(struct xdma_huge_mapping));
  if (rc != 0) {
    return rc;
  }

  if (map.npages < inst->huge_hnpages) {
    return -EFAULT;
  }

  map.npages = inst->huge_hnpages;
  int i = 0;
  int j = 0;
  int npages = inst->huge_npages;
  dma_addr = kmalloc(sizeof(unsigned long*) * map.npages, GFP_KERNEL);
  for (i; i < npages; i++) { //TODO += 512
    if (i % 512 == 0) {
      //printk(KERN_INFO "dma_addr: %p\n", page_to_phys(inst->huge_pages[i]));
      dma_addr[j] = page_to_phys(inst->huge_pages[i]);
      j++;
    }
  }
  printk(KERN_INFO "copy to user size: %i\n", sizeof(struct xdma_huge_mapping));

  if (copy_to_user((struct xdma_huge_mapping*)arg, &map, sizeof(struct xdma_huge_mapping))) {
    goto error;
  }

  if (copy_to_user(map.dma_addr, dma_addr, sizeof(unsigned long*) * map.npages)) {
    goto error;
  }

  kfree(dma_addr);
  return 0;

error:
  kfree(dma_addr);
  return -EACCES;
}

static int ioctl_release_mapping(struct dev_inst *inst)
{
  int rc;
  printk(KERN_INFO "IOCTL_XDMA_RELEASE");
  int npages = inst->huge_npages;
  int i = 0;
  int j = 0;
  for (i; i < npages; i++) {
    unsigned long pg = inst->huge_pages[i];
    printk(KERN_INFO "release dma_addr: %lx\n", pg);
    down_read(&current->mm->mmap_sem);
    if (!PageReserved(pg)) {
      SetPageDirty(pg);
    }
    put_page(pg);
    up_read(&current->mm->mmap_sem);

  }
  //dealloc pages
  vfree(inst->huge_pages);

  return 0;
}

static long char_xdma_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
  int rc = 0;
  struct dev_inst *inst;

  inst = (struct dev_inst*)file->private_data;
  printk(KERN_INFO "device inst ptr: %p\n", inst);

  switch (cmd) {
    case IOCTL_XDMA_BUFFER_SET:
      rc = ioctl_do_buffer_set(inst, arg);
      break;
    case IOCTL_XDMA_MAPPING_GET:
      rc = ioctl_do_mapping_get(inst, arg);
      break;
    case IOCTL_XDMA_RELEASE:
      rc = ioctl_release_mapping(inst);
      break;
    default:
      printk(KERN_INFO "char_xdma_ioctl() = %d, invalid cmd?\n", cmd);
      break;
  }

  return rc;
}

/*****************************/
/* The PCI driver            */
/*****************************/

/* This is mostly based on the Xilinx-supplied driver. */

/* The possible PCI device IDs of the Xilinx DMA core. */
static const struct pci_device_id pci_ids[] = {
    { PCI_DEVICE(0x10ee, 0x9011), },
    { PCI_DEVICE(0x10ee, 0x9012), },
    { PCI_DEVICE(0x10ee, 0x9014), },
    { PCI_DEVICE(0x10ee, 0x9018), },
    { PCI_DEVICE(0x10ee, 0x901F), },
    { PCI_DEVICE(0x10ee, 0x9021), },
    { PCI_DEVICE(0x10ee, 0x9022), },
    { PCI_DEVICE(0x10ee, 0x9024), },
    { PCI_DEVICE(0x10ee, 0x9028), },
    { PCI_DEVICE(0x10ee, 0x902F), },
    { PCI_DEVICE(0x10ee, 0x9031), },
    { PCI_DEVICE(0x10ee, 0x9032), },
    { PCI_DEVICE(0x10ee, 0x9034), },
    { PCI_DEVICE(0x10ee, 0x9038), },
    { PCI_DEVICE(0x10ee, 0x903F), },
    { PCI_DEVICE(0x10ee, 0x8011), },
    { PCI_DEVICE(0x10ee, 0x8012), },
    { PCI_DEVICE(0x10ee, 0x8014), },
    { PCI_DEVICE(0x10ee, 0x8018), },
    { PCI_DEVICE(0x10ee, 0x8021), },
    { PCI_DEVICE(0x10ee, 0x8022), },
    { PCI_DEVICE(0x10ee, 0x8024), },
    { PCI_DEVICE(0x10ee, 0x8028), },
    { PCI_DEVICE(0x10ee, 0x8031), },
    { PCI_DEVICE(0x10ee, 0x8032), },
    { PCI_DEVICE(0x10ee, 0x8034), },
    { PCI_DEVICE(0x10ee, 0x8038), },
    { PCI_DEVICE(0x10ee, 0x7011), },
    { PCI_DEVICE(0x10ee, 0x7012), },
    { PCI_DEVICE(0x10ee, 0x7014), },
    { PCI_DEVICE(0x10ee, 0x7018), },
    { PCI_DEVICE(0x10ee, 0x7021), },
    { PCI_DEVICE(0x10ee, 0x7022), },
    { PCI_DEVICE(0x10ee, 0x7024), },
    { PCI_DEVICE(0x10ee, 0x7028), },
    { PCI_DEVICE(0x10ee, 0x7031), },
    { PCI_DEVICE(0x10ee, 0x7032), },
    { PCI_DEVICE(0x10ee, 0x7034), },
    { PCI_DEVICE(0x10ee, 0x7038), },
    {0,}
};
MODULE_DEVICE_TABLE(pci, pci_ids);

struct interrupt_regs {
    u32 identifier;
    u32 user_int_enable;
    u32 user_int_enable_w1s;
    u32 user_int_enable_w1c;
    u32 channel_int_enable;
    u32 channel_int_enable_w1s;
    u32 channel_int_enable_w1c;
    u32 reserved_1[9];  /* padding */

    u32 user_int_request;
    u32 channel_int_request;
    u32 user_int_pending;
    u32 channel_int_pending;
    u32 reserved_2[12]; /* padding */

    u32 user_msi_vector[8];
    u32 channel_msi_vector[8];
} __packed;

struct config_regs {
    u32 identifier;
    u32 reserved_1[4];
    u32 msi_enable;
};

static struct dev_inst *
alloc_dev_instance(struct pci_dev *pdev)
{
    struct dev_inst *inst;

    BUG_ON(!pdev);

    /* allocate zeroed device book keeping structure */
    inst = kzalloc(sizeof(struct dev_inst), GFP_KERNEL);
    if (!inst) {
        printk(KERN_INFO "Could not kzalloc(dev_inst).\n");
        return NULL;
    }

    inst->config_bar_idx = -1;
    inst->user_bar_idx = -1;
    inst->bypass_bar_idx = -1;
    inst->irq_line = -1;
    //inst->buffer_created= false;
    inst->open_count= 0;
    inst->desc_bypass_enabled = false;
    //inst->transfer_running= 0;

    /* Create a device to driver reference */
    dev_set_drvdata(&pdev->dev, inst);
    /* create a driver to device reference */
    inst->pci_dev = pdev;

    printk(KERN_INFO "probe() inst = 0x%p\n", inst);

    return inst;
}

#ifndef arch_msi_check_device
int
arch_msi_check_device(struct pci_dev *dev, int nvec, int type) {
    return 0;
}
#endif

/* type = PCI_CAP_ID_MSI or PCI_CAP_ID_MSIX */
static int
msi_msix_capable(struct pci_dev *dev, int type) {
    struct pci_bus *bus;
    int ret;

    if (!dev || dev->no_msi) return 0;

    for (bus = dev->bus; bus; bus = bus->parent) {
        if (bus->bus_flags & PCI_BUS_FLAGS_NO_MSI) return 0;
    }

    ret = arch_msi_check_device(dev, 1, type);
    if (ret) return 0;

    if (!pci_find_capability(dev, type)) return 0;

    return 1;
}

static int
probe_scan_for_msi(struct dev_inst *inst, struct pci_dev *pdev) {
    int i;
    int rc = 0;
    int req_nvec = MAX_NUM_ENGINES + MAX_USER_IRQ;

    BUG_ON(!inst);
    BUG_ON(!pdev);

    if (msi_msix_capable(pdev, PCI_CAP_ID_MSIX)) {
        printk(KERN_INFO "Enabling MSI-X\n");
        for (i = 0; i < req_nvec; i++)
            inst->entry[i].entry = i;

        rc = pci_enable_msix_exact(pdev, inst->entry, req_nvec);
        if (rc < 0)
            printk(KERN_INFO "Couldn't enable MSI-X mode: rc = %d\n", rc);

        inst->msix_enabled = 1;
    } else if (msi_msix_capable(pdev, PCI_CAP_ID_MSI)) {
        /* enable message signalled interrupts */
        printk(KERN_INFO "pci_enable_msi()\n");
        rc = pci_enable_msi(pdev);
        if (rc < 0)
            printk(KERN_INFO "Couldn't enable MSI mode: rc = %d\n", rc);
        inst->msi_enabled = 1;
    } else {
        printk(KERN_INFO "MSI/MSI-X not detected - using legacy interrupts\n");
    }

    return rc;
}

static int
request_regions(struct dev_inst *inst, struct pci_dev *pdev) {
    int rc;

    BUG_ON(!inst);
    BUG_ON(!pdev);

    printk(KERN_INFO "pci_request_regions()\n");
    rc = pci_request_regions(pdev, DRV_NAME);
    /* could not request all regions? */
    if (rc) {
        printk(KERN_INFO "pci_request_regions() = %d, device in use?\n", rc);
        /* assume device is in use so do not disable it later */
        inst->regions_in_use = 1;
    } else {
        inst->got_regions = 1;
    }

    return rc;
}

static int
map_single_bar(struct dev_inst *inst, struct pci_dev *dev, int idx) {
    resource_size_t bar_start, bar_len, map_len;

    bar_start= pci_resource_start(dev, idx);
    bar_len=   pci_resource_len(dev, idx);
    map_len=   bar_len;

    inst->bar[idx] = NULL;

    /* do not map BARs with length 0. Note that start MAY be 0! */
    if(!bar_len) {
        printk(KERN_INFO "BAR #%d is not present - skipping\n", idx);
        return 0;
    }

    /* BAR size exceeds maximum desired mapping? */
    if(bar_len > INT_MAX) {
        printk(KERN_INFO "Limit BAR %d mapping from %llu to %d bytes\n",
               idx, (u64)bar_len, INT_MAX);
        map_len= (resource_size_t)INT_MAX;
    }
    /* Map the full device memory or IO region into kernel virtual
     * address space. */
    printk(KERN_INFO "BAR%d: %llu bytes to be mapped.\n",
           idx, (u64)map_len);
    inst->bar[idx] = pci_iomap(dev, idx, map_len);

    if (!inst->bar[idx]) {
        printk(KERN_INFO "Could not map BAR %d", idx);
        return -1;
    }

    printk(KERN_INFO "BAR%d at 0x%llx mapped at 0x%p, length=%llu(/%llu)\n",
           idx, (u64)bar_start, inst->bar[idx], (u64)map_len, (u64)bar_len);

    return (int)map_len;
}

static void
identify_bars(struct dev_inst *inst, int *bar_id_list, int num_bars,
              int config_bar_pos) {
    /* The following logic identifies which BARs contain what functionality
     * based on the position of the XDMA config BAR and the number of BARs
     * detected. The rules are that the user logic and bypass logic BARs are
     * optional.  When both are present, the XDMA config BAR will be the 2nd
     * BAR detected (config_bar_pos = 1), with the user logic being detected
     * first and the bypass being detected last. When one is omitted, the type
     * of BAR present can be identified by whether the XDMA config BAR is
     * detected first or last.  When both are omitted, only the XDMA config
     * BAR is present.  This somewhat convoluted approach is used instead of
     * relying on BAR numbers in order to work correctly with both 32-bit and
     * 64-bit BARs. */

    BUG_ON(!inst);
    BUG_ON(!bar_id_list);

    switch (num_bars) {
    case 1:
        /* Only one BAR present - no extra work necessary */
        break;

    case 2:
        if (config_bar_pos == 0) {
            inst->bypass_bar_idx = bar_id_list[1];
        } else if (config_bar_pos == 1) {
            inst->user_bar_idx = bar_id_list[0];
        } else {
            printk(KERN_INFO "case 2\n");
            printk(KERN_INFO "XDMA config BAR in unexpected position (%d)",
                   config_bar_pos);
        }
        break;

    case 3:
        if (config_bar_pos == 1) {
            inst->user_bar_idx = bar_id_list[0];
            inst->bypass_bar_idx = bar_id_list[2];
        } else {
            printk(KERN_INFO "case 3\n");
            printk(KERN_INFO "XDMA config BAR in unexpected position (%d)",
                   config_bar_pos);
        }
        break;

    default:
        /* Should not occur - warn user but safe to continue */
        printk(KERN_INFO "Unexpected number of BARs (%d)\n", num_bars);
        printk(KERN_INFO "Only XDMA config BAR accessible\n");
        break;
    }
}

/*
 * Unmap the BAR regions that had been mapped earlier using map_bars()
 */
static void
unmap_bars(struct dev_inst *inst, struct pci_dev *dev) {
    int i;

    for(i = 0; i < XDMA_BAR_NUM; i++) {
        /* is this BAR mapped? */
        if(inst->bar[i]) {
            /* unmap BAR */
            pci_iounmap(dev, inst->bar[i]);
            /* mark as unmapped */
            inst->bar[i] = NULL;
        }
    }
}

static int
is_config_bar(struct dev_inst *inst, int idx) {
    u32 irq_id = 0;
    u32 cfg_id = 0;
    int flag = 0;
    u32 mask = 0xffff0000; /* Compare only XDMA ID's not Version number */
    struct interrupt_regs *irq_regs =
        (struct interrupt_regs *) (inst->bar[idx] + XDMA_OFS_INT_CTRL);
    struct config_regs *cfg_regs =
        (struct config_regs *)(inst->bar[idx] + XDMA_OFS_CONFIG);

    irq_id = ioread32(&irq_regs->identifier);
    cfg_id = ioread32(&cfg_regs->identifier);

    if ((irq_id & mask) == IRQ_BLOCK_ID &&
        (cfg_id & mask) == CONFIG_BLOCK_ID) {
        printk(KERN_INFO "BAR %d is the XDMA config BAR\n", idx);
        flag = 1;
    } else {
        printk(KERN_INFO
               "BAR %d is not XDMA config BAR, irq_id = %x, cfg_id = %x\n",
               idx, irq_id, cfg_id);
        flag = 0;
    }

    return flag;
}

/* map_bars() -- map device regions into kernel virtual address space
 *
 * Map the device memory regions into kernel virtual address space after
 * verifying their sizes respect the minimum sizes needed
 */
static int
map_bars(struct dev_inst *inst, struct pci_dev *dev) {
    int rc;
    int i;
    int bar_id_list[XDMA_BAR_NUM];
    int bar_id_idx= 0;
    int config_bar_pos= 0;

    /* iterate through all the BARs */
    for(i= 0; i < XDMA_BAR_NUM; i++) {
        int bar_len;

        bar_len = map_single_bar(inst, dev, i);
        if(bar_len == 0) {
            continue;
        } else if(bar_len < 0) {
            rc = -1;
            goto fail;
        }

        /* Try to identify BAR as XDMA control BAR */
        if((bar_len >= XDMA_BAR_SIZE) && (inst->config_bar_idx < 0)) {
            if (is_config_bar(inst, i)) {
                inst->config_bar_idx = i;
                config_bar_pos = bar_id_idx;
            }
        }

        bar_id_list[bar_id_idx] = i;
        bar_id_idx++;
    }

    /* The XDMA config BAR must always be present */
    if(inst->config_bar_idx < 0) {
        printk(KERN_INFO "Failed to detect XDMA config BAR\n");
        rc = -1;
        goto fail;
    }

    identify_bars(inst, bar_id_list, bar_id_idx, config_bar_pos);

    /* successfully mapped all required BAR regions */
    return 0;
fail:
    /* unwind; unmap any BARs that we did map */
    unmap_bars(inst, dev);
    return rc;
}

int xdma_open(struct inode *inode, struct file *file) {
   struct dev_inst *inst;

   inst = container_of(inode->i_cdev, struct dev_inst, cdev);
   file->private_data = inst;

   return 0;
}

int xdma_release(struct inode *inode, struct file *file) {
   return 0;
}

static int
set_dma_mask(struct pci_dev *pdev) {
    int rc;

    BUG_ON(!pdev);

    printk(KERN_INFO "sizeof(dma_addr_t) == %ld\n", sizeof(dma_addr_t));
    rc= dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
    if(rc) printk(KERN_INFO "Couldn't set 64b DMA mask.\n");

    return rc;
}

/* channel_interrupts_enable -- Enable interrupts we are interested in */
static void
channel_interrupts_enable(struct dev_inst *inst, u32 mask) {
    struct interrupt_regs *reg = (struct interrupt_regs *)
        (inst->bar[inst->config_bar_idx] + XDMA_OFS_INT_CTRL);

    iowrite32(mask, &reg->channel_int_enable_w1s);
}

/* channel_interrupts_disable -- Disable interrupts we not interested in */
static void
channel_interrupts_disable(struct dev_inst *inst, u32 mask) {
    struct interrupt_regs *reg = (struct interrupt_regs *)
        (inst->bar[inst->config_bar_idx] + XDMA_OFS_INT_CTRL);

    iowrite32(mask, &reg->channel_int_enable_w1c);
}

/* read_interrupts -- Print the interrupt controller status */
static u32
read_interrupts(struct dev_inst *inst) {
    struct interrupt_regs *reg = (struct interrupt_regs *)
        (inst->bar[inst->config_bar_idx] + XDMA_OFS_INT_CTRL);
    u32 lo, hi;

    /* extra debugging; inspect complete engine set of registers */
    hi = ioread32(&reg->user_int_request);
    printk(KERN_INFO "ioread32(0x%p) returned 0x%08x (user_int_request).\n",
           &reg->user_int_request, hi);
    lo = ioread32(&reg->channel_int_request);
    printk(KERN_INFO
           "ioread32(0x%p) returned 0x%08x (channel_int_request)\n",
           &reg->channel_int_request, lo);

    /* return interrupts: user in upper 16-bits, channel in lower 16-bits */
    return build_u32(hi, lo);
}

static u32
build_vector_reg(u32 a, u32 b, u32 c, u32 d) {
    u32 reg_val = 0;

    reg_val |= (a & 0x1f) << 0;
    reg_val |= (b & 0x1f) << 8;
    reg_val |= (c & 0x1f) << 16;
    reg_val |= (d & 0x1f) << 24;

    return reg_val;
}

static void
write_msix_vectors(struct dev_inst *inst) {
    struct interrupt_regs *int_regs;
    u32 reg_val;

    BUG_ON(!inst);
    int_regs =(struct interrupt_regs *)
        (inst->bar[inst->config_bar_idx] + XDMA_OFS_INT_CTRL);

    /* user irq MSI-X vectors */
    reg_val = build_vector_reg(0, 1, 2, 3);
    iowrite32(reg_val, &int_regs->user_msi_vector[0]);

    reg_val = build_vector_reg(4, 5, 6, 7);
    iowrite32(reg_val, &int_regs->user_msi_vector[1]);

    reg_val = build_vector_reg(8, 9, 10, 11);
    iowrite32(reg_val, &int_regs->user_msi_vector[2]);

    reg_val = build_vector_reg(12, 13, 14, 15);
    iowrite32(reg_val, &int_regs->user_msi_vector[3]);

    /* channel irq MSI-X vectors */
    reg_val = build_vector_reg(16, 17, 18, 19);
    iowrite32(reg_val, &int_regs->channel_msi_vector[0]);

    reg_val = build_vector_reg(20, 21, 22, 23);
    iowrite32(reg_val, &int_regs->channel_msi_vector[1]);
}

static int
msix_irq_setup(struct dev_inst *inst) {
    BUG_ON(!inst);
    write_msix_vectors(inst);

    return 0;
}

static u32
get_engine_channel_id(struct engine_regs *regs) {
    u32 value;

    BUG_ON(!regs);

    value = ioread32(&regs->identifier);
    return (value & 0x00000f00U) >> 8;
}

static u32
get_engine_id(struct engine_regs *regs) {
    u32 value;

    BUG_ON(!regs);

    value = ioread32(&regs->identifier);
    return (value & 0xffff0000U) >> 16;
}

static int
probe_for_engine(struct dev_inst *inst, int dir_to_dev, int channel) {
    struct engine_regs *regs;
    int dir_from_dev;
    int offset;
    u32 engine_id;
    u32 engine_id_expected;
    u32 channel_id;
    int rc = 0;

    dir_from_dev = !dir_to_dev;

    /* register offset for the engine */
    /* read channels at 0x0000, write channels at 0x1000,
     * channels at 0x100 interval */
    offset = (dir_from_dev * H2C_CHANNEL_OFFSET) + (channel * CHANNEL_SPACING);

    regs = inst->bar[inst->config_bar_idx] + offset;
    if (dir_to_dev) {
        printk(KERN_INFO "Probing for H2C %d engine at %p\n", channel, regs);
        engine_id_expected = XDMA_ID_H2C;
    } else {
        printk(KERN_INFO "Probing for C2H %d engine at %p\n", channel, regs);
        engine_id_expected = XDMA_ID_C2H;
    }

    engine_id = get_engine_id(regs);
    channel_id = get_engine_channel_id(regs);
    printk(KERN_INFO "engine ID = 0x%x\n", engine_id);
    printk(KERN_INFO "engine channel ID = 0x%x\n", channel_id);

    if (engine_id != engine_id_expected) {
        printk(KERN_INFO "Incorrect engine ID - skipping\n");
        return 0;
    }

    if (channel_id != channel) {
        printk(KERN_INFO "Expected ch ID%d, read %d\n", channel, channel_id);
        return 0;
    }

    if (dir_to_dev) {
        inst->h2c_count++;
        printk(KERN_INFO "Found H2C %d AXI engine at %p\n", channel, regs);
    }
    else {
        inst->c2h_count++;
        printk(KERN_INFO "Found C2H %d AXI engine at %p\n", channel, regs);
    }

    if(dir_from_dev) {
        if(inst->c2h_engine == -1) {
            printk(KERN_INFO "Using C2H engine %d.\n", channel);
            inst->c2h_engine = channel;
        } else {
            printk(KERN_INFO "Ignoring C2H engine %d.\n", channel);
            return 0;
        }
    } else {
        if (inst->h2c_engine == -1) {
          printk(KERN_INFO "Using C2H engine %d.\n", channel);
          inst->h2c_engine = channel;
        } else {
          printk(KERN_INFO "Ignoring H2C engine %d.\n", channel);
          return 0;
        }
    }

    /* allocate and initialize engine */
    printk(KERN_INFO "XXX Enable engine here.\n"); /* XXX */

    return rc;
}

static int
probe_engines(struct dev_inst *inst) {
    int channel, h2c_offset, c2h_offset;//, sgdma_offset;
    u32 reg_value;

    BUG_ON(!inst);

    inst->h2c_count= 0;
    inst->c2h_count = 0;
    inst->h2c_engine= -1;
    inst->c2h_engine= -1;

    for(channel = 0; channel < XDMA_CHANNEL_NUM_MAX; channel++) {
        int rc;
        rc= probe_for_engine(inst, 1, channel);
        if(rc) return rc;
        rc= probe_for_engine(inst, 0, channel);
        if(rc) return rc;
    }

    if(inst->h2c_engine == -1 || inst->c2h_engine == -1) {
        printk(KERN_ERR "No C2H or H2C DMA engines found.\n");
        return -ENODEV;
    }

    printk(KERN_INFO "Found %d H2C DMA engines.\n", inst->h2c_count);
    printk(KERN_INFO "Found %d C2H DMA engines.\n", inst->c2h_count);

    //inst->engine_irq_mask= BIT(inst->h2c_count + inst->trace_engine);
    //printk(KERN_INFO "Channel IRQ mask is %08x.\n", inst->engine_irq_mask);

    h2c_offset = H2C_CHANNEL_OFFSET + (inst->h2c_engine * CHANNEL_SPACING);
    c2h_offset = C2H_CHANNEL_OFFSET + (inst->c2h_engine * CHANNEL_SPACING);
    //sgdma_offset = offset + SGDMA_OFFSET_FROM_CHANNEL;
    inst->h2c_regs = inst->bar[inst->config_bar_idx] + h2c_offset;
    inst->c2h_regs = inst->bar[inst->config_bar_idx] + c2h_offset;

    //inst->sgdma_regs = inst->bar[inst->config_bar_idx] + sgdma_offset;

    /* Enable descritpor bypass */
    printk(KERN_INFO "iowrite32(0x%08x to 0x%p) (control)\n", 0x1,
            (void *)&inst->h2c_regs->control);
    iowrite32(0x1, &inst->h2c_regs->control);
    printk(KERN_INFO "iowrite32(0x%08x to 0x%p) (control)\n", 0x1,
            (void *)&inst->c2h_regs->control);
    iowrite32(0x1, &inst->c2h_regs->control);
    inst->desc_bypass_enabled = true;

    /* Enable error interrupts. */
    /*reg_value = XDMA_CTRL_IE_DESC_ALIGN_MISMATCH;
    reg_value |= XDMA_CTRL_IE_MAGIC_STOPPED;
    reg_value |= XDMA_CTRL_IE_MAGIC_STOPPED;
    reg_value |= XDMA_CTRL_IE_READ_ERROR;
    reg_value |= XDMA_CTRL_IE_DESC_ERROR;*/

    /* Enable stopped, completed and idle interrupts. */
    /*reg_value |= XDMA_CTRL_IE_DESC_STOPPED;
    reg_value |= XDMA_CTRL_IE_DESC_COMPLETED;
    reg_value |= XDMA_CTRL_IE_IDLE_STOPPED;*/

    /* Clear all pending events. */
    //iowrite32(reg_value, &inst->regs->status);

    mmiowb();

    /* Enable interrupts. */
    //iowrite32(reg_value, &inst->regs->interrupt_enable_mask);

    return 0;
}

static void
irq_teardown(struct dev_inst *inst) {
    //int i;

    BUG_ON(!inst);

    if (inst->msix_enabled) {
#if 0
        for (i = 0; i < MAX_USER_IRQ; i++) {
            printk(KERN_INFO "Releasing IRQ#%d\n", inst->entry[i].vector);
            free_irq(inst->entry[i].vector, &inst->user_irq[i]);
        }
#endif
    } else if (inst->irq_line != -1) {
        printk(KERN_INFO "Releasing IRQ#%d\n", inst->irq_line);
        free_irq(inst->irq_line, inst);
    }
}

static int
probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    int rc = 0;
    struct dev_inst *inst = NULL;

    printk(KERN_INFO "probe(pdev = 0x%p, pci_id = 0x%p)\n", pdev, id);

    /* allocate zeroed device book keeping structure */
    inst = alloc_dev_instance(pdev);
    if(!inst) goto err_alloc;
    printk(KERN_INFO "Instance private data at %p\n", inst);

    rc = pci_enable_device(pdev);
    if(rc) {
        printk(KERN_INFO "pci_enable_device() failed, rc = %d.\n", rc);
        goto err_enable;
    }

    printk(KERN_INFO "device node %p\n", &pdev->dev);

    /* enable bus master capability */
    printk(KERN_INFO "pci_set_master()\n");
    pci_set_master(pdev);

    rc = probe_scan_for_msi(inst, pdev);
    if(rc < 0) goto err_scan_msi;

    rc = request_regions(inst, pdev);
    if(rc) goto err_regions;

    rc = map_bars(inst, pdev);
    if(rc) goto err_map;

    rc = set_dma_mask(pdev);
    if(rc) goto err_mask;

    //rc = irq_setup(inst, pdev);
    //if(rc) goto err_interrupts;

    rc = probe_engines(inst);
    if(rc) goto err_engines;

    /* Setup char device */
    //rc = setup_xdma_char(inst);
    //if(rc) goto err_char;

    /* Clear pending events. */
    printk(KERN_INFO "iowrite32(0x%08x to 0x%p) (status)\n", 0xFFFFFFFF,
            (void *)&inst->h2c_regs->status);
    iowrite32(0xFFFFFFFF, &inst->h2c_regs->status);
    printk(KERN_INFO "iowrite32(0x%08x to 0x%p) (status)\n", 0xFFFFFFFF,
            (void *)&inst->c2h_regs->status);
    iowrite32(0xFFFFFFFF, &inst->c2h_regs->status);

    /* Enable interrupts from the one channel we're using. */
    //channel_interrupts_enable(inst, inst->engine_irq_mask);
    //channel_interrupts_enable(inst, BIT(inst->trace_engine));

    /* Flush writes */
    //read_interrupts(inst);

    /* Create /dev/xdma_control. */
    misc_dev_node.minor= MISC_DYNAMIC_MINOR;
    misc_dev_node.name= "xdma_control";
    misc_dev_node.fops= &xdma_fops;
    misc_dev_node.parent= &pdev->dev;
    printk(KERN_INFO "Creating /dev/xdma_control (%p)\n", &misc_dev_node);
    rc= misc_register(&misc_dev_node);
    if(rc < 0) {
        printk(KERN_ERR "Failed to create /dev/xdma_control.\n");
        goto err_misc_register;
    }

    /* Create /dev/xdma_bypass. */
    bypass_node.minor= MISC_DYNAMIC_MINOR;
    bypass_node.name= "xdma_bypass";
    bypass_node.fops= &xdma_bypass_fops;
    bypass_node.parent= &pdev->dev;
    printk(KERN_INFO "Creating /dev/xdma_bypass (%p)\n", &bypass_node);
    rc= misc_register(&bypass_node);
    if(rc < 0) {
        printk(KERN_ERR "Failed to create /dev/xdma_bypass.\n");
        goto err_misc_register;
    }

    /* Add sysfs attributes for control and status. */
    /*dev_set_drvdata(misc_dev_node.this_device, inst);
    rc= sysfs_create_groups(&misc_dev_node.this_device->kobj, attr_groups);
    if(rc < 0) {
        printk(KERN_ERR "Failed to add sysfs attributes.\n");
        goto err_sysfs_create;
    }*/

    if (rc == 0) goto end;

/*err_sysfs_create:
    channel_interrupts_disable(inst, 0xFFFFFFFF);
    read_interrupts(inst);
    misc_deregister(&misc_dev_node);*/
err_misc_register:
err_char:
err_engines:
    irq_teardown(inst);
err_interrupts:
err_mask:
    unmap_bars(inst, pdev);
err_map:
    if(inst->got_regions) pci_release_regions(pdev);
err_regions:
    if(inst->msi_enabled) pci_disable_msi(pdev);
err_scan_msi:
    if(!inst->regions_in_use) pci_disable_device(pdev);
err_enable:
    kfree(inst);
err_alloc:
end:

    printk(KERN_INFO "probe() returning %d\n", rc);
    return rc;
}

static void
remove(struct pci_dev *pdev) {
    struct dev_inst *inst;

    printk(KERN_INFO "remove(0x%p)\n", pdev);
    if((pdev == 0) || (dev_get_drvdata(&pdev->dev) == 0)) {
        printk(KERN_INFO "remove(dev = 0x%p) pdev->dev.driver_data = 0x%p\n",
               pdev, dev_get_drvdata(&pdev->dev));
        return;
    }
    inst = (struct dev_inst *)dev_get_drvdata(&pdev->dev);
    printk(KERN_INFO
           "remove(dev = 0x%p) where pdev->dev.driver_data = 0x%p\n",
           pdev, inst);
    if(inst->pci_dev != pdev) {
        printk(KERN_INFO
               "pdev->dev.driver_data->pci_dev(0x%lx) != pdev(0x%lx)\n",
               (unsigned long)inst->pci_dev, (unsigned long)pdev);
    }

    //sysfs_remove_groups(&misc_dev_node.this_device->kobj, attr_groups);
    misc_deregister(&misc_dev_node);
    misc_deregister(&bypass_node);

    //channel_interrupts_disable(inst, 0xFFFFFFFF);
    //read_interrupts(inst);

    //irq_teardown(inst);
    unmap_bars(inst, pdev);

    if(inst->got_regions) pci_release_regions(pdev);

    if(inst->msix_enabled) {
        pci_disable_msix(pdev);
        inst->msix_enabled = 0;
    } else if (inst->msi_enabled) {
        pci_disable_msi(pdev);
        inst->msi_enabled = 0;
    }

    if(!inst->regions_in_use) pci_disable_device(pdev);

    kfree(inst);
}

static struct pci_driver pci_driver = {
    .name = DRV_NAME,
    .id_table = pci_ids,
    .probe = probe,
    .remove = remove,
};

static int __init
xdma_driver_init(void) {
    int rc = 0;
    //int i;

    rc = pci_register_driver(&pci_driver);

#if 0
    for(i=0;i<MAX_XDMA_DEVICES;i++){
        dev_present[i] = 0;
    }
#endif

    return rc;
}

static void __exit
xdma_driver_exit(void) {
    printk(KERN_INFO DRV_NAME" exit()\n");
    /* unregister this driver from the PCI bus driver */
    pci_unregister_driver(&pci_driver);
}

module_init(xdma_driver_init);
module_exit(xdma_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David Sidler <david.sidler@inf.ethz.ch");
