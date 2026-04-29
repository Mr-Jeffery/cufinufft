#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cufinufft_eitherprec.h>
#include "../contrib/utils.h"

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr,
            "Usage:\n"
            "  %s dense_real_with_header.txt nf1 nf2\n\n"
            "Args:\n"
            "  dense_real_with_header.txt : Dense file with header 'N1 N2', then N1*N2 values\n",
            argv[0]);
    return 1;
  }

  const std::string input_path = argv[1];

  std::vector<CUCPX> h_in;
  std::vector<CUCPX> h_out;
  std::vector<size_t> nz_idx;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float milliseconds = 0.0f;
  float totaltime = 0.0f;

  int N1 = 0, N2 = 0;
  int nf1 = 0, nf2 = 0;
  size_t n_elem_in = 0;

  auto file_start = std::chrono::high_resolution_clock::now();
  {
    std::ifstream fin(input_path.c_str());
    if (!fin.is_open()) {
      fprintf(stderr, "err: failed to open dense input file: %s\n", input_path.c_str());
      return 1;
    }

    if (!(fin >> N1 >> N2)) {
      fprintf(stderr,
              "err: failed to read dense header from %s\n"
              "     expected first line/header with two integers: N1 N2\n",
              input_path.c_str());
      return 1;
    }
    nf1 = (int)next235beven(N1, 1)*2;
    nf2 = (int)next235beven(N2, 1)*2;

    n_elem_in = (size_t)N1 * (size_t)N2;
    const size_t n_elem_fft = (size_t)nf1 * (size_t)nf2;

    h_in.assign(n_elem_fft, CUCPX({0.0, 0.0}));

    nz_idx.reserve(n_elem_in / 64 + 1);

    for (size_t i = 0; i < n_elem_in; ++i) {
      double v = 0.0;
      if (!(fin >> v)) {
        fprintf(stderr,
                "err: failed to read dense entry %zu from %s\n"
                "     expected %zu entries for header N1=%d N2=%d\n",
                i, input_path.c_str(), n_elem_in, N1, N2);
        return 1;
      }

      const int i1 = (int)(i / (size_t)N2);
      const int i2 = (int)(i % (size_t)N2);
      const size_t idx_fft = (size_t)i1 * (size_t)nf2 + (size_t)i2;

      h_in[idx_fft] = CUCPX({(FLT)v, (FLT)0.0});

      if (v != 0.0) nz_idx.push_back(idx_fft);
    }
  }
  auto file_end = std::chrono::high_resolution_clock::now();
  float file_ms = (float)std::chrono::duration_cast<std::chrono::microseconds>(file_end - file_start).count() / 1000.0f;
  totaltime += file_ms;
  printf("[time  ] file load:\t\t %.3g s\n", (double)file_ms / 1000.0);

  const size_t n_elem = (size_t)nf1 * (size_t)nf2;
  const size_t n_bytes = n_elem * sizeof(CUCPX);

  printf("[input ] loaded %zu dense entries from %s (%dx%d, format=dense-real-rowmajor+header)\n",
         n_elem_in, input_path.c_str(), N1, N2);
  printf("[input ] fft grid: %dx%d\n", nf1, nf2);
  printf("[input ] nonzero entries: %zu\n", nz_idx.size());

  cudaEventRecord(start);
  {
    cufftHandle warmup_plan;
    cufftPlan1d(&warmup_plan, 1, CUFFT_TYPE, 1);
    cufftDestroy(warmup_plan);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", (double)milliseconds / 1000.0);

  auto setup_start = std::chrono::high_resolution_clock::now();
  CUCPX *d_data = nullptr;
  cudaMalloc(&d_data, n_bytes);
  h_out.resize(n_elem);
  auto setup_end = std::chrono::high_resolution_clock::now();
  float setup_ms = (float)std::chrono::duration_cast<std::chrono::microseconds>(setup_end - setup_start).count() / 1000.0f;
  totaltime += setup_ms;
  printf("[time  ] host/device setup:\t %.3g s\n", (double)setup_ms / 1000.0);

  cufftHandle plan = 0;
  cudaEventRecord(start);
  cufftPlan2d(&plan, nf1, nf2, CUFFT_TYPE);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double plan_s = (double)milliseconds / 1000.0;
  totaltime += milliseconds;
  printf("[time  ] cufft plan:\t\t %.3g s\n", plan_s);

  cudaEventRecord(start);
  cudaMemcpy(d_data, h_in.data(), n_bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double set_s = (double)milliseconds / 1000.0;
  totaltime += milliseconds;
  printf("[time  ] cufft setUpts:\t\t %.3g s\n", set_s);

  cudaEventRecord(start);
  CUFFT_EX(plan, d_data, d_data, CUFFT_FORWARD);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  const double exec_s = (double)milliseconds / 1000.0;
  float exec_ms = milliseconds;
  totaltime += milliseconds;
  printf("[time  ] cufft exec:\t\t %.3g s\n", exec_s);

  cudaEventRecord(start);
  cufftDestroy(plan);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double destroy_s = (double)milliseconds / 1000.0;
  totaltime += milliseconds;

  cudaEventRecord(start);
  cudaMemcpy(h_out.data(), d_data, n_bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double copy_back_s = (double)milliseconds / 1000.0;
  totaltime += milliseconds;

  printf("[time  ] cufft destroy:\t\t %.3g s\n", destroy_s);
  printf("[time  ] cuda memcpy d2h:\t\t %.3g s\n", copy_back_s);


	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2);  // choose some mode index to check

  CPX Ft = CPX((FLT)0.0, (FLT)0.0);

  for (const size_t idx : nz_idx) {
    const int i1 = (int)(idx / (size_t)nf2);
    const int i2 = (int)(idx % (size_t)nf2);
    const FLT phase = (FLT)(-2.0) * (FLT)PI *
                      ((FLT)nt1 * (FLT)i1 / (FLT)nf1 + (FLT)nt2 * (FLT)i2 / (FLT)nf2);
    Ft += CPX((FLT)h_in[idx].x, (FLT)h_in[idx].y) * exp(IMA * phase);
  }

  const size_t it = (size_t)nt1 * (size_t)nf2 + (size_t)nt2;
  const CPX Fg = CPX((FLT)h_out[it].x, (FLT)h_out[it].y);

  const double rel_err = (double)(abs(Ft - Fg) / (abs(Ft) + (FLT)1e-30));

  const double total_s = (double)totaltime / 1000.0;
  const double nu_pts = (double)nz_idx.size();
  const double throughput = (total_s > 0.0) ? (nu_pts / total_s) : 0.0;
  const double exec_throughput = (exec_ms > 0.0f) ? (nu_pts / ((double)exec_ms / 1000.0)) : 0.0;

  printf("[Method 0] %zu NU pts to %zu U pts in %.3g s:      %.3g NU pts/s\n",
         nz_idx.size(), n_elem, total_s, throughput);
  printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n", exec_throughput);
  printf("[gpu   ] one mode: rel err in F[%d,%d] is %.3g\n", nt1, nt2, rel_err);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_data);

  return 0;
}
