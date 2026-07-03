#include "cudaHeader.cuh"

#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"

HD double Pincell::DTC(vec3 localPos, Neutron& n, XSLibrary* XSLib, GnuAMCM& RNG) {
	MatType matType = this->meatOrMod(localPos);
	MatXS& xs = XSLib->returnMatXSByType(matType);
	int g = static_cast<int>(n.energy);

	double sigC = collisionXS(xs, g);

	//double transXS = XSLib->returnMatXSByType(matType).transXS[static_cast<int>(n.energy)-1];
	return -log(RNG.uniform_open(0.0, 1.0)) / sigC;
}


HD double Pincell::DTS(vec3 localPos, Neutron& n) {
	//double epsT = 1.0e-13;

	// distance to the surfaces
	double distanceToWall = 1.0e300; // guard value
	double epsT = 0.0;
	if (n.dirVec.x >= epsT) {
		double tX = (this->sideLength - localPos.x) / n.dirVec.x;
		if (tX >= 0 && tX < distanceToWall) {
			distanceToWall = tX;
		}
	}
	else if (n.dirVec.x < -epsT) {
		double tX = (0.0 - localPos.x) / n.dirVec.x;
		if (tX >= 0 && tX < distanceToWall) {
			distanceToWall = tX;
		}
	}

	if (n.dirVec.y >= epsT) {
		double tY = (this->sideLength - localPos.y) / n.dirVec.y;
		if (tY >= 0 && tY < distanceToWall) {
			distanceToWall = tY;
		}
	}
	else if (n.dirVec.y < -epsT) {
		double tY = (0.0 - localPos.y) / n.dirVec.y;
		if (tY >= 0 && tY < distanceToWall) {
			distanceToWall = tY;
		}
	}

	if (n.dirVec.z >= epsT) {
		double tZ = (this->height - localPos.z) / n.dirVec.z;
		if (tZ >= 0 && tZ < distanceToWall) {
			distanceToWall = tZ;
		}
	}
	else if (n.dirVec.z < -epsT) {
		double tZ = (0.0 - localPos.z) / n.dirVec.z;
		if (tZ >= 0 && tZ < distanceToWall) {
			distanceToWall = tZ;
		}
	}

	double distanceToPin = 1.0e300;

	if (this->radius > 0.0) {
		vec3 localCenter = {
			this->sideLength / 2.0 + this->centerOffset.x,
			this->sideLength / 2.0 + this->centerOffset.y,
			this->height / 2.0
		};

		vec3 p = localPos - localCenter;

		double a = n.dirVec.x * n.dirVec.x + n.dirVec.y * n.dirVec.y;

		if (a > 1.0e-20) {
			double b = 2.0 * (p.x * n.dirVec.x + p.y * n.dirVec.y);
			double c = p.x * p.x + p.y * p.y - this->radius * this->radius;
			double disc = b * b - 4.0 * a * c;

			if (disc >= 0.0) {
				double s = sqrt(disc);
				double inv2a = 0.5 / a;

				double t0 = (-b - s) * inv2a;
				double t1 = (-b + s) * inv2a;

				if (t0 > t1) {
					double tmp = t0;
					t0 = t1;
					t1 = tmp;
				}

				if (t0 > epsT) distanceToPin = t0;
				else if (t1 > epsT) distanceToPin = t1;
			}
		}
	}


	// x^2 + y^2 = R^2 -> P^2 - R^2 = 0, where P = \vec{O} + t\vec{D}  (O: neutron localPos(offsetted), D: neutron directional vector (normalized))
	// implicit function P^2 - R^2 = 0 -> D^2t^2 + 2 O D t + O^2 - R^2 = 0, our target is t (the distance)
	// quadratic formula: since D^2 =1 (since normalized), t = OD \pm \sqrt{ (OD)^2 - O^2 + R^2 }
	// note that this only applies when the radius is not zero: for moderator blocks we should exclude this.
	/*
	double distanceToPin = 1.0e300;
	if (this->radius > 0.0) {
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
	}*/
	
	// choose the smaller value 
	/*
	if (distanceToWall >= distanceToPin)
	{
		
		if (distanceToPin < 1.0e-9) {
			double eps = 1.0e-3;
			double x = n.dirVec.x > 0 ? n.dirVec.x * eps : n.dirVec.x * -eps;
			double y = n.dirVec.y > 0 ? n.dirVec.y * eps : n.dirVec.y * -eps;
			double z = n.dirVec.z > 0 ? n.dirVec.z * eps : n.dirVec.z * -eps;
			vec3 epsVec = { x, y, z };
			n.pos = n.pos + epsVec;
		}
		
		
		return distanceToPin;
	}
	else {
		return distanceToWall;
	}
	*/
	return distanceToWall < distanceToPin ? distanceToWall : distanceToPin;
	
	
	/*
	double distanceToPin = 1.0e300;

	if (this->radius > 0.0) {
		vec3 localCenter = {
			this->sideLength / 2.0 + this->centerOffset.x,
			this->sideLength / 2.0 + this->centerOffset.y,
			this->height / 2.0
		};

		vec3 p = localPos - localCenter;

		constexpr double epsDir = 1.0e-14;
		constexpr double epsDist = 1.0e-12;
		constexpr double epsBoundary = 1.0e-10;

		double R2 = this->radius * this->radius;
		double r2 = p.x * p.x + p.y * p.y;

		double a = n.dirVec.x * n.dirVec.x + n.dirVec.y * n.dirVec.y;

		if (a > 1.0e-20) {
			double b = 2.0 * (p.x * n.dirVec.x + p.y * n.dirVec.y);
			double c = r2 - R2;
			double radialDir = p.x * n.dirVec.x + p.y * n.dirVec.y;

			if (fabs(c) <= epsBoundary) {
				if (fabs(radialDir) > epsDir) {
					distanceToPin = 0.0;
				}
			}
			else {
				double disc = b * b - 4.0 * a * c;

				if (disc > 1.0e-14) {
					double s = sqrt(disc);
					double inv2a = 0.5 / a;

					double t0 = (-b - s) * inv2a;
					double t1 = (-b + s) * inv2a;

					if (t0 > t1) {
						double tmp = t0;
						t0 = t1;
						t1 = tmp;
					}

					if (t0 > epsDist) {
						distanceToPin = t0;
					}
					else if (t1 > epsDist) {
						distanceToPin = t1;
					}
				}
			}
		}
	}
	return distanceToWall < distanceToPin ? distanceToWall : distanceToPin;
	*/
}

HD int Assembly::totalPincellNo() {
	return this->xNum * this->yNum * this->zNum;
}

HD Pincell& Assembly::returnPincellByIndex(int x, int y, int z) {
	if (x >= this->xNum || y >= this->yNum || z >= this->zNum) {
		printf("index [%d][%d][%d] out of bounds\n", x, y, z);
		//Pincell OOBPincell = Pincell(0, 0, 0);
		printf("OOB Pincell returned\n");
		return this->OOBPincell;
		//return this->pinCells[0];
	}
	
	int index = z * (this->xNum * this->yNum) + (this->xNum * y) + x;
	return this->pinCells[index];
}

//요걸좀고쳐보자
HD Pincell& Assembly::returnPincellByPos(const Neutron& n) {
	vec3 localAssemblyPos = n.pos - this->startPos;

	if (this->length.x == 0 || this->length.y == 0 || this->length.z == 0) {
		return this->OOBPincell;
	}
	
	//double eps = 1.0e-14;
	/*
	double x = n.dirVec.x > 0 ? eps : -eps;
	double y = n.dirVec.y > 0 ? eps : -eps;
	double z = n.dirVec.z > 0 ? eps : -eps;
	
	vec3 epsVec = { x, y, z };
	*/
	//localAssemblyPos = localAssemblyPos + epsVec;
		
	//n.pos = n.pos + epsVec;
	
	double cellSideLength = this->pinCells[0].sideLength;
	double cellHeight = this->pinCells[0].height;
	int xIdx = static_cast<int>(floor(localAssemblyPos.x / cellSideLength));
	int yIdx = static_cast<int>(floor(localAssemblyPos.y / cellSideLength));
	int zIdx = static_cast<int>(floor(localAssemblyPos.z / cellHeight));

	/* 
	if (xIdx == this->xNum || yIdx == this->yNum || zIdx == this->zNum) {
	}
	*/

	if (xIdx >= this->xNum || yIdx >= this->yNum || zIdx >= this->zNum || xIdx < 0 || yIdx < 0 || zIdx < 0) {
		printf("index [%d][%d][%d] out of bounds, neutron pos: (%f, %f, %f)\n", xIdx, yIdx, zIdx, n.pos.x, n.pos.y, n.pos.z);
		
		printf("OOB Pincell returned\n");
		return this->OOBPincell;
		//return this->pinCells[0];
	}

	int index = zIdx * (this->xNum * this->yNum) + (this->xNum * yIdx) + xIdx;
	return this->pinCells[index];
	
	//return this->returnPincellByIndex(xIdx, yIdx, zIdx);
}
	
HD vec3 Assembly::returnFlooredNeutronPosInPincell(const Neutron& n) {
	vec3 localAssemblyPos = n.pos - this->startPos;

	//vec3 cellLength = this->length / (this->xNum, this->yNum, this->zNum);

	double xLength = this->length.x / this->xNum;
	double yLength = this->length.y / this->yNum;
	double zLength = this->length.z / this->zNum;
	
	/*
	int xIdx = static_cast<int>((localAssemblyPos.x) / xLength);
	int yIdx = static_cast<int>((localAssemblyPos.y) / yLength);
	int zIdx = static_cast<int>((localAssemblyPos.z) / zLength);
	*/
	int xIdx = static_cast<int>(floor(localAssemblyPos.x / xLength));
	int yIdx = static_cast<int>(floor(localAssemblyPos.y / yLength));
	int zIdx = static_cast<int>(floor(localAssemblyPos.z / zLength));

	if (this->xNum <= 0 || this->yNum <= 0 || this->zNum <= 0 ||
		this->length.x <= 0.0 || this->length.y <= 0.0 || this->length.z <= 0.0) {
		return { 0.0, 0.0, 0.0 };
	}
	
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