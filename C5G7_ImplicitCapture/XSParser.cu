#include "cudaHeader.cuh"
#include "XSParser.cuh"




H void MatXS::g7DeviceAllocator(MatXS*& d_G7) {
	
	cudaMalloc(&d_G7, sizeof(MatXS));
	cudaMemcpy(d_G7, this, sizeof(MatXS), cudaMemcpyHostToDevice);

	/*

	double* d_totalXS = nullptr;
	double* d_transXS = nullptr;
	double* d_absXS = nullptr;
	double* d_capXS = nullptr;
	double* d_fisXS = nullptr;
	double* d_nu = nullptr;
	double* d_chi = nullptr;

	// This, you declare a pointer to 7 arrays.
	double (*d_elsXS)[7] = nullptr;

	cudaMalloc(&d_totalXS, sizeof(double) * 7);
	cudaMalloc(&d_transXS, sizeof(double) * 7);
	cudaMalloc(&d_absXS, sizeof(double) * 7);
	cudaMalloc(&d_capXS, sizeof(double) * 7);
	cudaMalloc(&d_fisXS, sizeof(double) * 7); 
	cudaMalloc(&d_nu, sizeof(double) * 7);
	cudaMalloc(&d_chi, sizeof(double) * 7);
	cudaMalloc(&d_elsXS, sizeof(this->elsXS));



	cudaMemcpy(d_totalXS, &(this->totalXS), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_transXS, &(this->transXS), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_absXS, &(this->absXS), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_capXS, &(this->capXS), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fisXS, &(this->fisXS), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nu, &(this->nu), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_chi, &(this->chi), sizeof(double) * 7, cudaMemcpyHostToDevice);
	cudaMemcpy(d_elsXS, this->elsXS, 
	

	*/

}