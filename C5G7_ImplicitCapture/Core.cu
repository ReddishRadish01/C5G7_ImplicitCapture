#include "cudaHeader.cuh"

#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"

HD double Pincell::DTC(vec3 localPos, Neutron& n, XSLibrary* XSLib, GnuAMCM& RNG) {
	MatType matType = this->meatOrMod(localPos);
	
	double transXS = XSLib->returnMatXSByType(matType).transXS[static_cast<int>(n.energy)-1];
	return -log(RNG.uniform(0.0, 1.0)) / transXS;
}


HD double Pincell::DTS(vec3 localPos, Neutron& n) {
	double epsT = 1.0e-13;

	// distance to the surfaces
	double distanceToWall = 1.0e300; // guasrd value
	if (n.dirVec.x > epsT) {
		double tX = (this->sideLength - localPos.x) / n.dirVec.x;
		if (tX >= 0 && tX < distanceToWall) {
			distanceToWall = tX;
		}
	}
	else if (n.dirVec.x < epsT) {
		double tX = (0.0 - localPos.x) / n.dirVec.x;
		if (tX >= 0 && tX < distanceToWall) {
			distanceToWall = tX;
		}
	}

	if (n.dirVec.y > epsT) {
		double tY = (this->sideLength - localPos.y) / n.dirVec.y;
		if (tY >= 0 && tY < distanceToWall) {
			distanceToWall = tY;
		}
	}
	else if (n.dirVec.y < epsT) {
		double tY = (0.0 - localPos.y) / n.dirVec.y;
		if (tY >= 0 && tY < distanceToWall) {
			distanceToWall = tY;
		}
	}

	if (n.dirVec.z > epsT) {
		double tZ = (this->height - localPos.z) / n.dirVec.z;
		if (tZ >= 0 && tZ < distanceToWall) {
			distanceToWall = tZ;
		}
	}
	else if (n.dirVec.z < epsT) {
		double tZ = (0.0 - localPos.z) / n.dirVec.z;
		if (tZ >= 0 && tZ < distanceToWall) {
			distanceToWall = tZ;
		}
	}

	
	// x^2 + y^2 = R^2 -> P^2 - R^2 = 0, where P = \vec{O} + t\vec{D}  (O: neutron localPos(offsetted), D: neutron directional vector (normalized))
	// implicit function P^2 - R^2 = 0 -> D^2t^2 + 2 O D t + O^2 - R^2 = 0, our target is t (the distance)
	// quadratic formula: since D^2 =1 (since normalized), t = OD \pm \sqrt{ (OD)^2 - O^2 + R^2 }
	// note that this only applies when the radius is not zero: for moderator blocks we should exclude this.
	double distanceToPin = 1.0e300;
	if (this->radius > epsT) {
		vec3 localCenter = { this->sideLength / 2.0 + this->centerOffset.x, this->sideLength / 2.0 + this->centerOffset.y, this->height / 2.0 };
		localPos = localPos - localCenter;

		double a = n.dirVec.x * n.dirVec.x + n.dirVec.y * n.dirVec.y;
		double b = 2.0 * (localPos.x * n.dirVec.x + localPos.y * n.dirVec.y);
		double c = localPos.x * localPos.x + localPos.y * localPos.y - this->radius * this->radius;
		double BsqAC_2 = b * b - 4 * a * c;
		double s = sqrt(BsqAC_2);
		double inv2a = 0.5 / a;

		double t0 = (-b - s) * inv2a;
		double t1 = (-b + s) * inv2a;

		if (t0 > t1) {
			double tmp = t0; t0 = t1; t1 = tmp;
		}

		if (t0 > epsT) { distanceToPin = t0; }
		else if (t1 > epsT) { distanceToPin = t1; }
	}
	
	// choose the smaller value 
	if (distanceToWall >= distanceToPin)
	{
		return distanceToPin;
	}
	else {
		return distanceToWall;
	}
	

	//return distanceToWall;
}

HD int Assembly::totalPincellNo() {
	return this->xNum * this->yNum * this->zNum;
}

HD Pincell& Assembly::returnPincellByIndex(int x, int y, int z) {
	if (x >= this->xNum || y >= this->yNum || z >= this->zNum) {
		printf("index [%d][%d][%d] out of bounds\n", x, y, z);
		Pincell OOBPincell = Pincell(0, 0, 0);
		printf("OOB Pincell returned\n");
		return OOBPincell;
		//return this->pinCells[0];
	}
	
	int index = z * (this->xNum * this->yNum) + (this->xNum * y) + x;
	return this->pinCells[index];
}

HD Pincell& Assembly::returnPincellByPos(Neutron& n) {
	vec3 localAssemblyPos = n.pos - this->startPos;
	
	double eps = 1.0e-14;
	
	double x = n.dirVec.x > 0 ? eps : -eps;
	double y = n.dirVec.y > 0 ? eps : -eps;
	double z = n.dirVec.z > 0 ? eps : -eps;
	
	vec3 epsVec = { x, y, z };
	
	//localAssemblyPos = localAssemblyPos + epsVec;
		
	//n.pos = n.pos + epsVec;
	
	double cellSideLength = this->pinCells[0].sideLength;
	double cellHeight = this->pinCells[0].height;
	int xIdx = static_cast<int>((localAssemblyPos.x ) / cellSideLength);
	int yIdx = static_cast<int>((localAssemblyPos.y ) / cellSideLength);
	int zIdx = static_cast<int>((localAssemblyPos.z ) / cellHeight);

	if (xIdx == this->xNum || yIdx == this->yNum || zIdx == this->zNum) {
		
	}
	
	if (xIdx >= this->xNum || yIdx >= this->yNum || zIdx >= this->zNum) {
		printf("index [%d][%d][%d] out of bounds, neutron pos: (%f, %f, %f)\n", xIdx, yIdx, zIdx, n.pos.x, n.pos.y, n.pos.z);
		
		printf("OOB Pincell returned\n");
		return this->OOBPincell;
		//return this->pinCells[0];
	}

	int index = zIdx * (this->xNum * this->yNum) + (this->xNum * yIdx) + xIdx;
	return this->pinCells[index];
	
	//return this->returnPincellByIndex(xIdx, yIdx, zIdx);
}

HD vec3 Assembly::returnFlooredNeutronPosInPincell(Neutron& n) {
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


HD double Assembly::DTC(Neutron& n, XSLibrary* XSLib, GnuAMCM& RNG) {
	Pincell currentPincell = this->returnPincellByPos(n);
	vec3 pincellLocalPos = this->returnFlooredNeutronPosInPincell(n);
	//MatType mat = currentPincell.meatOrMod(pincellLocalPos);
	/*
	if (currentPincell.sideLength == 0.0 && currentPincell.height == 0.0) {
		n.Nullify();
		return 1.0e+300;
	}
	*/
	return currentPincell.DTC(pincellLocalPos, n, XSLib, RNG);
}

HD double Assembly::DTS(Neutron& n) {
	Pincell currentPincell = this->returnPincellByPos(n);
	vec3 pincellLocalPos = this->returnFlooredNeutronPosInPincell(n);

	return currentPincell.DTS(pincellLocalPos, n);
}