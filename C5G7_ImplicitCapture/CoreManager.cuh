#pragma once

#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"


// this is for finding location
class CoreManager {
public:
	HD static MatType getInteractionMaterial(C5G7Geometry core, Neutron& neutron) {

		// first find the position of the Pincell that is located.

		// and then, we will use member funciton of PinCell: meatOrMod - which returns a MatType.

		return MatType::Unknown;
	}

	HD inline static Pincell returnPincellByPos(C5G7Geometry* core, Neutron n) {
 		return core->returnAssemblyByNeutron(n).returnPincellByPos(n);
	}

};


