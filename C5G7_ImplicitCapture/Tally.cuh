#pragma once

#include "cudaHeader.cuh"
#include <cuda_runtime.h>
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "GpuManager.cuh"
#include "CoreManager.cuh"
#include "XSManager.cuh"
#include "NeutronBankManager.cuh"
#include "Debug.cuh"
#include <iomanip>


class TallyPincell {
public:
	double sideLength = 1.26;	// 1.26 cm. alias with height
	double height = 1.26;
	double radius = 0.54; 
	double pinTally = 0.0;
	double modTally = 0.0;

	H TallyPincell() = default;

	H TallyPincell(double sideLength, double radius = 0.0, double height = 0.0) 
		: sideLength(sideLength), radius(radius), height(height)
	{
		if (2 * radius > sideLength) { this->radius = sideLength / 2.0; }
	}
};

class TallyAssembly {
public:
	vec3 startPos = { 0, 0, 0 };

	vec3 length = { 0, 0, 0 };

	int xNum = 0;
	int yNum = 0;
	int zNum = 0;
	TallyPincell* pinCells = nullptr;
	TallyPincell OOBPincell;

	H TallyAssembly() = default;

	H void Initialize(std::string assemblyTxt, double cellHeight = 0.0, vec3 startPos = { 0, 0, 0 }, vec3 endPos = { 0, 0, 0 }) {

		std::ifstream assembly(assemblyTxt);

		if (!assembly.is_open()) {
			throw std::runtime_error("Cannot open: " + assemblyTxt);
		}

		if (!(assembly >> this->xNum >> this->yNum >> this->zNum)) {
			throw std::runtime_error("Failed to read xNum yNum zNum from: " + assemblyTxt);
		}

		//assembly >> this->xNum >> this->yNum >> this->zNum;
		int numPincells = this->xNum * this->yNum * this->zNum;

		double pincellLength = 0.0;
		double pincellHeight = 0.0;
		double radius = 0.0;

		assembly >> pincellLength >> pincellHeight >> radius;

		if (cellHeight != 0.0) {
			pincellHeight = cellHeight;
		}
		int mod = 0;
		assembly >> mod;

		MatType modType = (mod >= 0 && mod <= 6) ? static_cast<MatType>(mod) : MatType::MOD;

		this->pinCells = new TallyPincell[numPincells];

		for (int i = 0; i < xNum * yNum; i++) {
			for (int j = 0; j < zNum; j++) {
				int index = j * (xNum * yNum) + i;
				this->pinCells[index] = TallyPincell(pincellLength, radius, pincellHeight);
			}
			//std::cout << 
		}

		assembly.close();

		OOBPincell = TallyPincell(pincellLength, radius, pincellHeight);
	}

	H void Initialize_MOD(double cellSize, vec3 startPos = { 0, 0, 0 }, vec3 endPos = { 0, 0, 0 }, double cellHeight = 0.0) {
		vec3 length = endPos - startPos;

		double height = 0.0;
		if (cellHeight == 0.0) {
			height = cellSize;
		}
		else {
			height = cellHeight;
		}

		this->startPos = startPos;
		this->length = length;

		this->xNum = static_cast<int>((length.x + 1.0e-9) / cellSize);
		this->yNum = static_cast<int>((length.y + 1.0e-9) / cellSize);
		this->zNum = static_cast<int>((length.z + 1.0e-9) / height);
		// fuck this floating point shits - the length.z is not enough - 

		int numPincells = this->xNum * this->yNum * this->zNum;
		this->pinCells = new TallyPincell[numPincells];

		for (int i = 0; i < this->xNum * this->yNum; i++) {
			for (int j = 0; j < this->zNum; j++) {
				int index = j * (this->xNum * this->yNum) + i;
				this->pinCells[index] = TallyPincell(cellSize, 0, height);
				//std::cout << static_cast<int>(this->pinCells[j].meatType) << " ";
			}
		}

	}

	HD TallyPincell& returnPincellByPos(Neutron& n);
	HD vec3 returnFlooredNeutronPosInPincell(Neutron& n);

	HD int totalPincellNo() {
		return xNum * yNum * zNum;
	}
};

class TallyC5G7Geometry {
public:
	double x; // cm
	double y;
	double z;

	TallyAssembly* tallyAssembly = nullptr;
	TallyAssembly nullAssembly{};
	int assemblyNo = 0;

	TallyC5G7Geometry(C5G7Geometry Core) {
		this->x = Core.x; this->y = Core.y; this->z = Core.z;


		this->assemblyNo = Core.assemblyNo;
		tallyAssembly = new TallyAssembly[this->assemblyNo];
		for (int i = 0; i < this->assemblyNo; i++) {
			this->tallyAssembly[i].startPos = Core.assembly[i].startPos;
			this->tallyAssembly[i].length = Core.assembly[i].length;
			this->tallyAssembly[i].xNum = Core.assembly[i].xNum;
			this->tallyAssembly[i].yNum = Core.assembly[i].yNum;
			this->tallyAssembly[i].zNum = Core.assembly[i].zNum;
			int pinCellNo = this->tallyAssembly[i].xNum * this->tallyAssembly[i].yNum * this->tallyAssembly[i].zNum;
			this->tallyAssembly[i].pinCells = new TallyPincell[pinCellNo];
			for (int j = 0; j < pinCellNo; j++) {
				double sideLength = this->tallyAssembly[i].length.x / this->tallyAssembly[i].xNum;
				double height = this->tallyAssembly[i].length.z / this->tallyAssembly[i].zNum;
				this->tallyAssembly[i].pinCells[j].sideLength = sideLength;
				this->tallyAssembly[i].pinCells[j].height = height;
				this->tallyAssembly[i].pinCells[j].radius = 0.0;
			}
			this->tallyAssembly->OOBPincell = TallyPincell(1.26, 0.0, 1.26);
		}
	}
	

	HD TallyAssembly& returnAssemblyByNeutron(Neutron& n) {
		double eps = 1.0e-12;
		double x = n.dirVec.x > 0 ? n.dirVec.x * eps : n.dirVec.x * -eps;
		double y = n.dirVec.y > 0 ? n.dirVec.y * eps : n.dirVec.y * -eps;
		double z = n.dirVec.z > 0 ? n.dirVec.z * eps : n.dirVec.z * -eps;
		vec3 epsVec = { x, y, z };
		//Neutron localN = n;
		//localN.pos = localN.pos + epsVec;
		//n.pos = n.pos + epsVec;
		vec3 localPos = n.pos + epsVec;

		for (int i = 0; i < this->assemblyNo; i++) {
			vec3 endPos = this->tallyAssembly[i].startPos + this->tallyAssembly[i].length;
			if (localPos.x >= this->tallyAssembly[i].startPos.x && localPos.x < endPos.x) {
				if (localPos.y >= this->tallyAssembly[i].startPos.y && localPos.y < endPos.y) {
					if (localPos.z >= this->tallyAssembly[i].startPos.z && localPos.z < endPos.z) {
						return this->tallyAssembly[i];
					}
				}
			}
		}

		// this usually means fucked up - 
		// we are never meant to pass the out-of-bounds neutrons, or nullified neutrons in this function.
		// we never want the code to flow into this far, in this function - idK why it ended up here, maybe put some debugger outputs just in case
		//printf("Neutron Out-Of-Bounds in Position: (%f, %f, %f)\n", n.pos.x, n.pos.y, n.pos.z);
		//n.Nullify();
		// return null assembly 
		return this->nullAssembly;
	}
};

class Tally {
public:


	D inline static void kTally() {
		
	}



};


static inline void DumpCoreTallyToText(const TallyC5G7Geometry& h_CoreTally, std::ofstream& out, int cycleNo, int targetZCell, double k) {
	out << std::setprecision(17);
	out << "\n\n\nCoreSize(cm) " << h_CoreTally.x << " " << h_CoreTally.y << " " << h_CoreTally.z << "\n";
	out << "AssemblyNo " << h_CoreTally.assemblyNo << "\n";

	out << cycleNo << " th cycle:\n";
	out << k << " multiplication Factor\n";


	for (int a = 0; a < h_CoreTally.assemblyNo - 1; ++a) {
		const auto& Asm = h_CoreTally.tallyAssembly[a];
		const int nx = Asm.xNum;
		const int ny = Asm.yNum;
		const int nz = Asm.zNum;
		const int n = nx * ny * nz;

		out << "\n";
		out << "Assembly " << a << "\n";
		out << "StartPos " << Asm.startPos.x << " " << Asm.startPos.y << " " << Asm.startPos.z << "\n";
		out << "Length   " << Asm.length.x << " " << Asm.length.y << " " << Asm.length.z << "\n";
		out << "Dims     " << nx << " " << ny << " " << nz << "\n";
		//out << "OOB      " << Asm.OOBPincell.pinTally << " " << Asm.OOBPincell.modTally << "\n";
		out << "Format   a k j i pinTally modTally\n";

		

		for (int k = 0; k < nz; ++k) {
			if (k == targetZCell) {
				for (int j = 0; j < ny; ++j) {
					for (int i = 0; i < nx; ++i) {
						const int idx = k * (nx * ny) + j * nx + i;
						const auto& pc = Asm.pinCells[idx];
						out << a << " " << k << " " << j << " " << i << " " << pc.pinTally << " " << pc.modTally << "\n";
					}
				}
			}
		}
	}
}

__global__ void ZeroTallyPincellsKernel(TallyPincell* pcs, int n);

void ResetCoreTallyOnDevice(const TallyC5G7Geometry& h_CoreTally, TallyAssembly* d_bufferTallyAssembly, const std::vector<TallyPincell*>& d_bufferTallyPincellVec);