URL = http://yann.lecun.com/exdb/mnist/

PACKS = \
	train-images-idx3-ubyte \
	train-labels-idx1-ubyte \
	t10k-images-idx3-ubyte \
	t10k-labels-idx1-ubyte

PACKS_COMPILED = $(foreach PACK, ${PACKS}, ${PACK}.gz)


all: ${PACKS}

${PACKS}:
	$(foreach PACK, ${PACKS_COMPILED}, wget ${URL}${PACK};)
	gzip -d ${PACKS_COMPILED}

clean:
	$(foreach PACK, ${PACKS_COMPILED}, rm -f ${PACK})

delete: clean
	$(foreach PACK, ${PACKS}, rm -f ${PACK};)