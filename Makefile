SUBDIRS := kernel

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in "${SUBDIRS}"; do make -C "$${dir}" clean; done
	-rm -rf build
	-rm -f g6k/siever.cpp
	-rm -f g6k/siever.so

.PHONY: all $(SUBDIRS) clean python
