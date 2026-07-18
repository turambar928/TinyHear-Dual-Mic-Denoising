CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror

all: tiny_tcn_ref

tiny_tcn_ref: main.c tiny_tcn_int8.c tiny_tcn_int8.h realtime_dsp.c realtime_dsp.h fft_backend.c fft_backend.h generated/model_config.h generated/model_requant.h generated/test_vectors.h generated/realtime_vectors.h generated/band_matrix.h ../runs/arctic_demand/int8/model_int8.h
	$(CC) $(CFLAGS) main.c tiny_tcn_int8.c realtime_dsp.c fft_backend.c -lm -o tiny_tcn_ref

tiny_tcn_bench: benchmark.c tiny_tcn_int8.c tiny_tcn_int8.h realtime_dsp.c realtime_dsp.h fft_backend.c fft_backend.h generated/model_config.h generated/model_requant.h generated/test_vectors.h generated/realtime_vectors.h generated/band_matrix.h ../runs/arctic_demand/int8/model_int8.h
	$(CC) $(CFLAGS) benchmark.c tiny_tcn_int8.c realtime_dsp.c fft_backend.c -lm -o tiny_tcn_bench

run: tiny_tcn_ref
	./tiny_tcn_ref

bench: tiny_tcn_bench
	./tiny_tcn_bench

clean:
	$(RM) tiny_tcn_ref tiny_tcn_bench

.PHONY: all run bench clean
