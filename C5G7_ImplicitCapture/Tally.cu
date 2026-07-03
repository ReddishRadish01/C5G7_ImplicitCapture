#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "GpuManager.cuh"
#include "CoreManager.cuh"
#include "XSManager.cuh"
#include "NeutronBankManager.cuh"
#include "Debug.cuh"

#include "Tally.cuh"


HD TallyPincell& TallyAssembly::returnPincellByPos(Neutron& n) {
	vec3 localAssemblyPos = n.pos - this->startPos;

	double eps = 1.0e-11;
	double x = n.dirVec.x > 0 ? eps : -eps;
	double y = n.dirVec.y > 0 ? eps : -eps;
	double z = n.dirVec.z > 0 ? eps : -eps;
	vec3 epsVec = { x, y, z };
	localAssemblyPos = localAssemblyPos + epsVec;

	//n.pos = n.pos + epsVec;

	double cellSideLength = this->pinCells[0].sideLength;
	double cellHeight = this->pinCells[0].height;
	int xIdx = static_cast<int>(localAssemblyPos.x / cellSideLength);
	int yIdx = static_cast<int>(localAssemblyPos.y / cellSideLength);
	int zIdx = static_cast<int>(localAssemblyPos.z / cellHeight);


	if (xIdx >= this->xNum || yIdx >= this->yNum || zIdx >= this->zNum) {
		//printf("TallyPincell: index [%d][%d][%d] out of bounds, neutron pos: (%f, %f, %f). OOB Pincell returned. SL: %f, H: %f\n ", xIdx, yIdx, zIdx, n.pos.x, n.pos.y, n.pos.z, cellSideLength, cellHeight);
		return this->OOBPincell;
		//return this->pinCells[0];
	}

	int index = zIdx * (this->xNum * this->yNum) + (this->xNum * yIdx) + xIdx;
	return this->pinCells[index];

	//return this->returnPincellByIndex(xIdx, yIdx, zIdx);
}


HD vec3 TallyAssembly::returnFlooredNeutronPosInPincell(Neutron& n) {
	vec3 localAssemblyPos = n.pos - this->startPos;

	//vec3 cellLength = this->length / (this->xNum, this->yNum, this->zNum);

	double xLength = this->length.x / this->xNum;
	double yLength = this->length.y / this->yNum;
	double zLength = this->length.z / this->zNum;


	int xIdx = localAssemblyPos.x / xLength;
	int yIdx = localAssemblyPos.y / yLength;
	int zIdx = localAssemblyPos.z / zLength;


	return { localAssemblyPos.x - xLength * xIdx,
			 localAssemblyPos.y - yLength * yIdx,
			 localAssemblyPos.z - zLength * zIdx };
}

__global__ void ZeroTallyPincellsKernel(TallyPincell* pcs, int n) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < n) {
		pcs[t].pinTally = 0.0;
		pcs[t].modTally = 0.0;
	}
}

void ResetCoreTallyOnDevice(const TallyC5G7Geometry& h_CoreTally, TallyAssembly* d_bufferTallyAssembly, const std::vector<TallyPincell*>& d_bufferTallyPincellVec) {
	const int A = h_CoreTally.assemblyNo;

	for (int i = 0; i < A; i++) {
		const int n = h_CoreTally.tallyAssembly[i].totalPincellNo();
		int threads = 256;
		int blocks = (n + threads - 1) / threads;
		ZeroTallyPincellsKernel << <blocks, threads >> > (d_bufferTallyPincellVec[i], n);
		cudaDeviceSynchronize();
	}
}
