# CSC 766 Final Project

## Introduction
In this project, a binary sparse matrix (in COO format) dedicated FFT algorithm is implemented and compared with the dense implementation with `cufft`. This project is based on [`flatironinstitute/cufinufft`](https://github.com/flatironinstitute/cufinufft), an FFT algorithm that is designed for Non-Uniform inputs/outputs on 1D/2D/3D. The original implementation is able to utilize the sparsity of the input, but falls short on utilizing the binary nature for faster computation.

## Implementation & Modification
- Dedicated spread implementation for binary sparse matrix, which significantly reduced the matrix transition time by eliminating the unnecessary matrix value transcation. Notice how `cnow` is defined inside the kernel instead of fetching from `*c`. By using wrapper, this kernel will only be called if `*c` is passed as `nullptr`.
```c++
/* ------------------------ 2d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

__global__
void Spread_2d_NUptsdriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, int *idxnupts, 
		int pirange)
{
	int xstart,ystart,xend,yend;
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	FLT kervalue1, kervalue2;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		cnow = c[idxnupts[i]];

		xstart = ceil(x_rescaled - ns/2.0);
		ystart = ceil(y_rescaled - ns/2.0);
		xend = floor(x_rescaled + ns/2.0);
		yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		for(yy=ystart; yy<=yend; yy++){
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				kervalue1=ker1[xx-xstart];
				kervalue2=ker2[yy-ystart];
				atomicAdd(&fw[outidx].x, cnow.x*kervalue1*kervalue2);
				atomicAdd(&fw[outidx].y, cnow.y*kervalue1*kervalue2);
			}
		}

	}

}

/* Kernels for NUptsdriven Method with Binary Matrix*/

__global__
void Spread_2d_NUptsdriven_Bin(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, int *idxnupts, 
		int pirange)
{
	int xstart,ystart,xend,yend;
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	FLT kervalue1, kervalue2;

	assert(c == nullptr); // this kernel is only for testing spreading with binmtx, so c should be null and all values in fw should be added by 1.0
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);

		xstart = ceil(x_rescaled - ns/2.0);
		ystart = ceil(y_rescaled - ns/2.0);
		xend = floor(x_rescaled + ns/2.0);
		yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		for(yy=ystart; yy<=yend; yy++){
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				kervalue1=ker1[xx-xstart];
				kervalue2=ker2[yy-ystart];
				atomicAdd(&fw[outidx].x, kervalue1*kervalue2);
			}
		}

	}

}
```

## Installation & Run Guide
For `double` precision, run
```bash
make -j4 bin/cufinufft2d1_binmtx_test bin/cufft_dense2d_test
export CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH}"
./bin/cufinufft2d1_binmtx_test 1 benzene.mtx
./bin/cufft_dense2d_test benzene_dense_real.mtx
```
For `single` precision, run
```bash
make -j4 bin/cufinufft2d1_binmtx_test bin/cufft_dense2d_test
export CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH}"
./bin/cufinufft2d1_binmtx_test_32 1 benzene.mtx
./bin/cufft_dense2d_test_32 benzene_dense_real.mtx
```

## Performance Analysis
Benchmark environment:
- OS: Ubuntu 24.04.3 LTS on Windows 10 x86_64 under WSL2
- CPU: 12th Gen Intel i9-12900HX (24 cores)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- NVIDIA driver: 572.83
- CUDA version reported by `nvidia-smi`: 12.8

The benzene benchmark shows that the COO-based sparse path is faster end-to-end than the dense cuFFT baseline, mainly because it avoids materializing and transforming the full dense grid. The sparse path is strongest in total runtime, while the dense path is still slightly ahead in pure FFT execution throughput.

| Method | Precision | Total time | Exec-only throughput | Relative error |
| --- | --- | ---: | ---: | ---: |
| Sparse bin-matrix | Single | 0.752 s | 8.36e+05 NU pts/s | 8.49e-07 |
| Dense cuFFT | Single | 6.41 s | 9.59e+05 NU pts/s | 5.39e-04 |
| Sparse bin-matrix | Double | 2.9 s | 8.76e+04 NU pts/s | 3.24e-16 |
| Dense cuFFT | Double | 10.5 s | 1.21e+05 NU pts/s | 2.36e-12 |

From these runs, the sparse implementation is about 8.5x faster overall in single precision and about 3.6x faster overall in double precision. The dense path still reaches higher exec-only throughput in both precisions, which suggests the win for the sparse version comes primarily from lower data movement, reduced setup overhead, and less work before and after the FFT. See the measured runs in [complex32.txt](complex32.txt), [complex64.txt](complex64.txt), [dense32.txt](dense32.txt), and [dense64.txt](dense64.txt).

## Limitation & Unsuccessful Attempts
Due to limited time, there are some promising optimization direction:
- Changing the execution kernel `CUFFT_EX` to a customized kernel did not work, because the `cuFFTExecZ2Z()` and `cuFFTExecC2C()` are highly optimized, simply using a sparse binary FFT can not leverage the lack of optimization.
- Changing `cuFFTExecZ2Z()` and `cuFFTExecC2C()` to `cuFFTExecD2Z()` and `cuFFTExecR2C()` also did not work, which is against the claim on CUFFT documentation. Even though we can reduce the input to real matrix and drop half of the size, they are still slower than the complex-to-complex implementation. We assume this is due to the lack of optimization.