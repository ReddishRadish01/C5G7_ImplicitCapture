#pragma once
#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "GpuManager.cuh"
#include "CoreManager.cuh"
#include "XSManager.cuh"
#include "NeutronBankManager.cuh"
#include "Tally.cuh"


class Debug {

public:
	H static void fuelLayoutDebug(Assembly& assembly) {
		std::cout << "\nAssembly layout info: total number of pincells in this assembly: " << assembly.totalPincellNo() << "\n";
		std::cout << "start: (" << assembly.startPos.x << ", " << assembly.startPos.y << ", " << assembly.startPos.z << "). ";
		std::cout << "length: (" << assembly.length.x << ", " << assembly.length.y << ", " << assembly.length.z << ").\n";
		std::cout << "dimension: (" << assembly.xNum << ", " << assembly.yNum << ", " << assembly.zNum << ").\n";
		std::cout << "for the z=0 layer:\n";
		for (int k = 0; k < 1; k++) {
			for (int ii = 0; ii < assembly.yNum; ii++) {
				for (int ij = 0; ij < assembly.xNum; ij++) {
					int index = (ii * assembly.xNum + ij);
					std::cout << static_cast<int>(assembly.pinCells[index].meatType) << " ";
				}
				std::cout << "\n";
			}
		}
	}
	
};