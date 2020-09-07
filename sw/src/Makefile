
# note that pkg-config respects PKG_CONFIG_PATH, which can be used
# to pick libcheck from a user-defined path


COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lcuda -lpthread -ldl -lgdrapi

SRCS := bw.cpp
EXES := $(SRCS:.cpp=)

all: exes

exes: $(EXES)

bw.o: bw.cpp common.hpp


bw: bw.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

clean:
	rm -f *.o $(EXES) *~ core.*

# install: exes
# 	@ echo "installing exes in $(DESTBIN)..." && \
# 	mkdir -p $(DESTBIN) && \
# 	install -D -v -m u=rwx,g=rx,o=rx bw -t $(DESTBIN) && \

.PHONY: clean all exes install
