#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cctype>

#include <cufinufft_eitherprec.h> // commented so that the LSP can find the correct functions in cufinufft2d.cu instead of the macros here, will uncomment later when compiling

#include "../contrib/utils.h"

using namespace std;

static string to_lower(string s)
{
	for (size_t i = 0; i < s.size(); ++i) {
		s[i] = (char)tolower((unsigned char)s[i]);
	}
	return s;
}

static bool has_mtx_suffix(const string& s)
{
	const string suffix = ".mtx";
	if (s.size() < suffix.size()) return false;
	return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static int load_matrix_market_rows_cols(const string& path, int& nrows, int& ncols,
																				vector<int>& rows, vector<int>& cols,
																				string& field, string& symmetry)
{
	ifstream fin(path.c_str());
	if (!fin.is_open()) {
		fprintf(stderr, "err: failed to open Matrix Market file: %s\n", path.c_str());
		return 1;
	}

	string line;
	if (!getline(fin, line)) {
		fprintf(stderr, "err: empty Matrix Market file: %s\n", path.c_str());
		return 1;
	}

	istringstream hs(line);
	string banner, object, format;
	hs >> banner >> object >> format >> field >> symmetry;
	object = to_lower(object);
	format = to_lower(format);
	field = to_lower(field);
	symmetry = to_lower(symmetry);
	if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
		fprintf(stderr, "err: unsupported Matrix Market format in %s\n", path.c_str());
		return 1;
	}
	if (field != "real" && field != "integer" && field != "pattern" && field != "complex") {
		fprintf(stderr, "err: unsupported Matrix Market field '%s' in %s\n", field.c_str(), path.c_str());
		return 1;
	}

	int nnz = 0;
	while (getline(fin, line)) {
		if (line.empty() || line[0] == '%') continue;
		istringstream ds(line);
		if (!(ds >> nrows >> ncols >> nnz)) {
			fprintf(stderr, "err: failed to parse Matrix Market size line in %s\n", path.c_str());
			return 1;
		}
		break;
	}
	if (nrows <= 0 || ncols <= 0 || nnz <= 0) {
		fprintf(stderr, "err: invalid matrix dimensions/nnz in %s\n", path.c_str());
		return 1;
	}

	rows.reserve(nnz);
	cols.reserve(nnz);
	while (getline(fin, line)) {
		if (line.empty() || line[0] == '%') continue;
		istringstream es(line);
		int r = 0, c = 0;
		if (!(es >> r >> c)) continue;
		if (r < 1 || r > nrows || c < 1 || c > ncols) {
			fprintf(stderr, "err: out-of-range COO entry (%d, %d) in %s\n", r, c, path.c_str());
			return 1;
		}
		rows.push_back(r - 1);
		cols.push_back(c - 1);
	}

	if (rows.empty()) {
		fprintf(stderr, "err: no COO entries parsed from %s\n", path.c_str());
		return 1;
	}

	return 0;
}

int main(int argc, char* argv[])
{
	int N1 = 0, N2 = 0, M = 0, N = 0;
	if (argc<3) {
		fprintf(stderr,
			"Usage:\n"
			"  cufinufft2d1_binmtx_test method N1 N2 [M [tol]]\n"
			"  cufinufft2d1_binmtx_test method matrix.mtx [tol]\n"
			"  cufinufft2d1_binmtx_test method N1 N2 matrix.mtx [tol]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven,\n"
			"    2: sub-problem, or\n"
			"    3: sub-problem with Paul's idea.\n"
			"  N1, N2: The size of the 2D array.\n"
			"  M: The number of non-uniform points (default N1 * N2).\n"
			"  matrix.mtx: Matrix Market COO input (row/col used, values ignored).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);

	float totaltime = 0;

	FLT tol=1e-6;
	bool use_mtx = false;
	string mtx_path;
	vector<int> mtx_rows, mtx_cols;
	string mtx_field, mtx_symmetry;

	if (has_mtx_suffix(string(argv[2]))) {
		use_mtx = true;
		mtx_path = argv[2];
		if (argc > 3) {
			sscanf(argv[3], "%lf", &w);
			tol = (FLT)w;
		}
	} else {
		if (argc < 4) {
			fprintf(stderr, "err: expected N1 N2 or matrix.mtx\n");
			return 1;
		}
		sscanf(argv[2], "%lf", &w);
		N1 = (int)w;
		sscanf(argv[3], "%lf", &w);
		N2 = (int)w;

		if (argc > 4) {
			if (has_mtx_suffix(string(argv[4]))) {
				use_mtx = true;
				mtx_path = argv[4];
				if (argc > 5) {
					sscanf(argv[5], "%lf", &w);
					tol = (FLT)w;
				}
			} else {
				sscanf(argv[4], "%lf", &w);
				M = (int)w;
				if (argc > 5) {
					sscanf(argv[5], "%lf", &w);
					tol = (FLT)w;
				}
			}
		}
	}

	if (use_mtx) {
		auto load_start = chrono::high_resolution_clock::now();
		int file_n1 = 0, file_n2 = 0;
		int ier_load = load_matrix_market_rows_cols(
				mtx_path, file_n1, file_n2, mtx_rows, mtx_cols, mtx_field, mtx_symmetry);
		if (ier_load != 0) return ier_load;
		auto load_end = chrono::high_resolution_clock::now();
		float load_ms = (float)chrono::duration_cast<chrono::microseconds>(load_end - load_start).count() / 1000.0f;
		totaltime += load_ms;
		printf("[time  ] file load:\t\t %.3g s\n", load_ms / 1000.0f);

		if (N1 == 0 && N2 == 0) {
			N1 = file_n1;
			N2 = file_n2;
		} else if (N1 != file_n1 || N2 != file_n2) {
			fprintf(stderr,
							"err: N1/N2 (%d,%d) do not match matrix dimensions (%d,%d) in %s\n",
							N1, N2, file_n1, file_n2, mtx_path.c_str());
			return 1;
		}

		M = (int)mtx_rows.size();
		printf("[input ] loaded %d COO entries from %s (%dx%d, field=%s, symmetry=%s)\n",
					 M, mtx_path.c_str(), N1, N2, mtx_field.c_str(), mtx_symmetry.c_str());
	}

	if (N1 <= 0 || N2 <= 0) {
		fprintf(stderr, "err: invalid N1/N2 (%d, %d)\n", N1, N2);
		return 1;
	}
	if (!use_mtx && M <= 0) {
		M = N1 * N2;
	}
	if (M <= 0) {
		fprintf(stderr, "err: invalid M (%d)\n", M);
		return 1;
	}
	N = N1 * N2;
	int iflag=1;


	cout<<scientific<<setprecision(3);
	int ier;
	cudaEvent_t start, stop;
	float milliseconds = 0;


	auto setup_start = chrono::high_resolution_clock::now();
	FLT *x, *y;
	CPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	// cudaMallocHost(&c, M*sizeof(CPX)); // to be removed for binary matrix, since we won't have c values
	cudaMallocHost(&fk,N1*N2*sizeof(CPX));

	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	// checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX))); // to be removed for binary matrix, since we won't have c values
	d_c = nullptr; // binary matrix: no coefficients
	checkCudaErrors(cudaMalloc(&d_fk,N1*N2*sizeof(CUCPX)));

	// Making data: either random synthetic points or Matrix Market COO-derived points.
	for (int i = 0; i < M; i++) {
		if (use_mtx) {
			x[i] = (FLT)(-M_PI + (2.0 * M_PI * (double)mtx_rows[i]) / (double)N1);
			y[i] = (FLT)(-M_PI + (2.0 * M_PI * (double)mtx_cols[i]) / (double)N2);
			// c[i] = CPX((FLT)1.0, (FLT)0.0); // to be removed for binary matrix, since we won't have c values
		} else {
			x[i] = M_PI*randm11();// x in [-pi,pi)
			y[i] = M_PI*randm11();
			c[i].real(randm11());
			c[i].imag(randm11());
		}
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CUCPX),cudaMemcpyHostToDevice)); // to be removed for binary matrix, since we won't have c values
	auto setup_end = chrono::high_resolution_clock::now();
	float setup_ms = (float)chrono::duration_cast<chrono::microseconds>(setup_end - setup_start).count() / 1000.0f;
	totaltime += setup_ms;
	printf("[time  ] host/device setup:\t %.3g s\n", setup_ms / 1000.0f);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// warm up CUFFT (is slow, takes around 0.2 sec... )
	cudaEventRecord(start);
 	{
		int nf1=1;
		cufftHandle fftplan;
		cufftPlan1d(&fftplan,nf1,CUFFT_TYPE,1);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds/1000);

	// now to our tests...
	CUFINUFFT_PLAN dplan;
	int dim = 2;
	int type = 1;

	// Here we setup our own opts, for gpu_method.
	cufinufft_opts opts;
	ier=CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
	if(ier!=0){
	  printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
	  return ier;
	}

	opts.gpu_method=method;

	int nmodes[3];
	int ntransf = 1;
	int maxbatchsize = 1;
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;
	cudaEventRecord(start);
	// Left unchanged, does not concern d_c
	ier=CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol,
			       maxbatchsize, &dplan, &opts);
	if (ier!=0){
	  printf("err: cufinufft2d_plan\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);
	printf("[plan  ] nf1=%d nf2=%d\n", dplan->nf1, dplan->nf2);


	cudaEventRecord(start);
	// Left unchanged, does not concern d_c
	ier=CUFINUFFT_SETPTS(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, dplan);
	if (ier!=0){
	  printf("err: cufinufft_setpts\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	/** Needs to change so that it can take d_c as nullptr for binary matrix
	The call stack for our case:
	 CUFINUFFT_EXECUTE (cufinufft.cu)
	 	-> CUFINUFFT2D1_EXEC (cufinufft2d.cu)
	 		-> CUSPREAD2D (spread2d_wrappers.cu)
	 			-> CUSPREAD2D_NUPTSDRIVEN (spread2d_wrappers.cu)
					-> Spread_2d_NUptsdriven (spreadinterp2d.cu)
			-> CUFFT_EX (alias for cufftExecZ2Z or cufftExecC2C depending on precision) we will need no inplement a cufftExecB2C for binary matrix case
			-> CUDECONVOLVE2D (deconvolve_wrapper.cu)
				-> Deconvolve_2d (cufinufft2d.cu)
	*/
	ier=CUFINUFFT_EXECUTE(d_c, d_fk, dplan);
	if (ier!=0){
	  printf("err: cufinufft2d1_exec\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	float exec_ms = milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=CUFINUFFT_DESTROY(dplan);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	auto d2h_start = chrono::high_resolution_clock::now();
	checkCudaErrors(cudaMemcpy(fk,d_fk,N1*N2*sizeof(CUCPX),
		cudaMemcpyDeviceToHost));
	auto d2h_end = chrono::high_resolution_clock::now();
	float d2h_ms = (float)chrono::duration_cast<chrono::microseconds>(d2h_end - d2h_start).count() / 1000.0f;
	totaltime += d2h_ms;
	printf("[time  ] cuda memcpy d2h:\t %.3g s\n", d2h_ms / 1000.0f);

	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2);  // choose some mode index to check
	CPX Ft = CPX(0,0), J = IMA*(FLT)iflag;
	for (int j=0; j<M; ++j)
		Ft += (c == nullptr ? CPX(1.0, 0.0) : c[j]) * exp(J*(nt1*x[j]+nt2*y[j]));   // crude direct
	printf("[Method %d] %d NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n",
			opts.gpu_method,M,N1*N2,totaltime/1000,M/totaltime*1000);
	printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n",M/exec_ms*1000);
	int it = N1/2+nt1 + N1*(N2/2+nt2);   // index in complex F as 1d array
//	printf("[gpu   ] one mode: abs err in F[%ld,%ld] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it]));
	printf("[gpu   ] one mode: rel err in F[%d,%d] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it])/infnorm(N,fk));

	cudaFreeHost(x);
	cudaFreeHost(y);
	// cudaFreeHost(c);
	cudaFreeHost(fk);
	cudaFree(d_x);
	cudaFree(d_y);
	// cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
