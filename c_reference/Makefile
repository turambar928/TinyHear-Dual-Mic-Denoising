CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Werror
FFT_BACKEND ?= naive
CMSIS_CFLAGS ?=
CMSIS_LDLIBS ?=

ifeq ($(FFT_BACKEND),cmsis)
FFT_BACKEND_SRC := fft_backend_cmsis.c
FFT_BACKEND_CFLAGS := $(CMSIS_CFLAGS)
FFT_BACKEND_LDLIBS := $(CMSIS_LDLIBS)
else
FFT_BACKEND_SRC := fft_backend.c
FFT_BACKEND_CFLAGS :=
FFT_BACKEND_LDLIBS :=
endif

all: tiny_tcn_ref

tiny_tcn_ref: main.c tiny_tcn_int8.c tiny_tcn_int8.h tiny_gate.c tiny_gate.h realtime_dsp.c realtime_dsp.h $(FFT_BACKEND_SRC) fft_backend.h generated/model_config.h generated/model_requant.h generated/test_vectors.h generated/realtime_vectors.h generated/band_matrix.h generated/tiny_gate_params.h ../runs/arctic_demand_spatial_c116/int8/model_int8.h
	$(CC) $(CFLAGS) $(FFT_BACKEND_CFLAGS) main.c tiny_tcn_int8.c tiny_gate.c realtime_dsp.c $(FFT_BACKEND_SRC) -lm $(FFT_BACKEND_LDLIBS) -o tiny_tcn_ref

tiny_tcn_bench: benchmark.c tiny_tcn_int8.c tiny_tcn_int8.h tiny_gate.c tiny_gate.h realtime_dsp.c realtime_dsp.h $(FFT_BACKEND_SRC) fft_backend.h generated/model_config.h generated/model_requant.h generated/test_vectors.h generated/realtime_vectors.h generated/band_matrix.h generated/tiny_gate_params.h ../runs/arctic_demand_spatial_c116/int8/model_int8.h
	$(CC) $(CFLAGS) $(FFT_BACKEND_CFLAGS) benchmark.c tiny_tcn_int8.c tiny_gate.c realtime_dsp.c $(FFT_BACKEND_SRC) -lm $(FFT_BACKEND_LDLIBS) -o tiny_tcn_bench

run: tiny_tcn_ref
	./tiny_tcn_ref

bench: tiny_tcn_bench
	./tiny_tcn_bench

clean:
	$(RM) tiny_tcn_ref tiny_tcn_bench

.PHONY: all run bench clean
