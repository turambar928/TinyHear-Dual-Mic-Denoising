CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror

all: tiny_tcn_ref

tiny_tcn_ref: main.c tiny_tcn_int8.c tiny_tcn_int8.h realtime_dsp.c realtime_dsp.h generated/model_config.h generated/model_requant.h generated/test_vectors.h generated/realtime_vectors.h generated/band_matrix.h ../runs/arctic_demand/int8/model_int8.h
	$(CC) $(CFLAGS) main.c tiny_tcn_int8.c realtime_dsp.c -lm -o tiny_tcn_ref

run: tiny_tcn_ref
	./tiny_tcn_ref

clean:
	$(RM) tiny_tcn_ref

.PHONY: all run clean
