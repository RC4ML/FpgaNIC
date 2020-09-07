typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

typedef struct gdr *gdr_t;

struct gdr {
    int fd;
    LIST_HEAD(memh_list, gdr_memh_t) memhs;
};

typedef struct gdr_memh_t { 
    uint32_t handle;
    LIST_ENTRY(gdr_memh_t) entries;
    unsigned mapped:1;
    unsigned wc_mapping:1;
} gdr_memh_t;

LIST_ENTRY(gdr_memh_t) entries;
= struct { struct gdr_memh_t *le_next; struct gdr_memh_t **le_prev; } entries
// LIST_ENTRY(gdr_memh_t) entries;

LIST_HEAD(memh_list, gdr_memh_t) memhs;
= struct memh_list { struct gdr_memh_t *lh_first; } memhs

memh_list没有意义