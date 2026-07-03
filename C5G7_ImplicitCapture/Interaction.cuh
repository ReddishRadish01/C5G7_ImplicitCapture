#pragma once

#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "XSManager.cuh"
#include "CoreManager.cuh"

//#define INTERACTIONDEBUG 
//#define FISSIONPRINT



class Interaction {
public:
	HD static void scatter(Neutron& neutron, double outEnergy, GnuAMCM& RNG) {
		//neutron.updateWithLength(1.0e-10);
		neutron.energy = outEnergy;
		neutron.dirVec = vec3::randomUnit(RNG);
		//printf(" I SCATTER\n");
	}

	HD static void absorption(Neutron& n) {
		n.Nullify();
	}

	D static void fission(Neutron& n, NeutronBank* Bank, MatXS& matXS, GnuAMCM& RNG, double* k_mult, bool passFlag) {
		double nu = matXS.nu[static_cast<int>(n.energy) - 1];
		int fissionNum = static_cast<int>(nu / *k_mult + RNG.uniform(0.0, 1.0));

		n.dirVec = vec3::randomUnit(RNG);
		int fissionE = XSManager::returnFissionNeutronEnergy(matXS, RNG);
		int addIndex = atomicAdd(&(Bank->addedNeutronIndex), fissionNum - 1);
		int hardCap = addIndex + (fissionNum - 1);
		if (hardCap >= Bank->allocatableNeutronNum) {
			return;
		}

		atomicAdd(&(Bank->addedNeutronSize), fissionNum - 1);
		n.energy = fissionE;

		for (int i = 0; i < fissionNum - 1; i++) {
			//Bank->addedNeutrons[addIndex + i].status = true;
			int fissionE = XSManager::returnFissionNeutronEnergy(matXS, RNG);
			Bank->addedNeutrons[addIndex + i].reInitialize(n.pos, vec3::randomUnit(RNG), fissionE, 1.0, passFlag);
		}

	}

	D static void fission_forced(Neutron& n, NeutronBank* Bank, MatXS& matXS, GnuAMCM& RNG, double* k_mult, bool passFlag, double* fissionStrength, double& fissionCount) {
		
		int g = static_cast<int>(n.energy);
		double nu = matXS.nu[g - 1];
		if (nu == 0.0) { return; }

		double motherWeight = n.weight;
		
		double sigC = collisionXS(matXS, g);

		//
		double daughterWeight = n.weight * matXS.fisXS[g-1] / sigC;

		int fissionNum = static_cast<int>(nu / *k_mult * daughterWeight + RNG.uniform(0.0, 1.0));
		fissionCount = fissionNum;
		// for Fission source collision estimator, you have to sum the source weight, not the Bernoulli result of source weight.
		atomicAdd(fissionStrength, nu * daughterWeight);
		//printf("fissionNum: %d\n", fissionNum);
		//n.dirVec = vec3::randomUnit(RNG);
		int addIndex = 0;

		if (fissionNum <= 0) {
			//printf("No fission neutron added: \n");
			return;
		}
		else {
			//printf("New neutron added from forced fision: %d. Parent Neutron weight: %f \n", fissionNum, motherWeight);
			//if (fissionNum >= 2) { printf("fissionNum greater than 2: %d\n", fissionNum); }
			addIndex = atomicAdd(&(Bank->addedNeutronIndex), fissionNum);
			if (addIndex + fissionNum > Bank->allocatableNeutronNum) {
				printf("addIndex overflow: %d\n", addIndex);
				return;
			}
			atomicAdd(&(Bank->addedNeutronSize), fissionNum);
			for (int i = 0; i < fissionNum; i++) {
				int fissionE = XSManager::returnFissionNeutronEnergy(matXS, RNG);
				Bank->addedNeutrons[addIndex + i].reInitialize(n.pos, vec3::randomUnit(RNG), fissionE, 1.0, passFlag);
			}
		}

	}

	D static void russianRoulette(Neutron& n, NeutronBank* Bank, GnuAMCM& RNG, double weightCutoff, double newWeight, bool addTrue) {
		if (n.weight < weightCutoff) {
			double oldWeight = n.weight;
			double rouletteRNG = RNG.uniform(0.0, 1.0);
			double survivalProbability = n.weight / newWeight;
			if (rouletteRNG < survivalProbability) {
				//printf("Neutron survived RR with weight of %f\n", oldWeight);
				n.setWeight(newWeight);
			}
			else {
				n.Nullify();
				//printf("Neutron Nullified with weight of %f\n", oldWeight);
				if (addTrue == true) {
					atomicAdd(&(Bank->addedNeutronSize), -1);
				}
				else {
					atomicAdd(&(Bank->neutronSize), -1);
				}

			}
		}
	}

	D static void reaction_implicit(Neutron& n, NeutronBank* Bank, XSLibrary* XSLib, Pincell currentPincell, vec3 localPos, GnuAMCM& RNG, double* k_mult, bool passFlag, bool add, double* fissionWeight, double& fissionCount) {
		MatType currentMat = currentPincell.meatOrMod(localPos);
		double outEnergy = XSManager::scatteringEnergy(XSLib, currentMat, RNG, n.energy);
		Interaction::fission_forced(n, Bank, XSLib->returnMatXSByType(currentMat), RNG, k_mult, passFlag, fissionWeight, fissionCount);

		MatXS& xs = XSLib->returnMatXSByType(currentMat);
		int g = static_cast<int>(n.energy);
		double sigS = scatterSum(xs, g);
		double sigC = xs.capXS[g - 1] + xs.fisXS[g - 1] + sigS;

		n.setWeight(n.weight * sigS / sigC);

		// implicit capture: get only the elastic XS out of all the XS sum
		//double elasXS = XSLib->returnMatXSByType(currentMat).transXS[static_cast<int>(n.energy)-1] - XSLib->returnMatXSByType(currentMat).capXS[static_cast<int>(n.energy) - 1] - XSLib->returnMatXSByType(currentMat).fisXS[static_cast<int>(n.energy) - 1];
		//double elasXS = XSLib->returnMatXSByType(currentMat).transXS[static_cast<int>(n.energy) - 1] - XSLib->returnMatXSByType(currentMat).absXS[static_cast<int>(n.energy) - 1];
		//n.setWeight(n.weight * elasXS / XSLib->returnMatXSByType(currentMat).transXS[static_cast<int>(n.energy) - 1]);
		//InteractionType interactionT = XSManager::returnInteracitonType(XSLib, currentMat, RNG, n.energy, outEnergy);
		//printf("currentEnergy: %f, outEnergy: %f\n", n.energy, outEnergy);
		Interaction::scatter(n, outEnergy, RNG);
	}

	D static void reaction(Neutron& n, NeutronBank* Bank, XSLibrary* XSLib, Pincell currentPincell, vec3 localPos, GnuAMCM& RNG, double* k_mult, bool passFlag, bool add) {
		MatType currentMat = currentPincell.meatOrMod(localPos);
		double outEnergy = 0.0;
		Interaction::fission(n, Bank, XSLib->returnMatXSByType(currentMat), RNG, k_mult, passFlag);

		InteractionType interactionT = XSManager::returnInteracitonType(XSLib, currentMat, RNG, n.energy, outEnergy);
		if (interactionT == InteractionType::nel) {
			
			Interaction::scatter(n, outEnergy, RNG);
		}
		else if (interactionT == InteractionType::ng) {

			Interaction::absorption(n);
			if (add == true) { // absorption in addedneutron
				atomicAdd(&(Bank->addedNeutronSize), -1);
			}
			else {
				atomicAdd(&(Bank->neutronSize), -1);
			}
		}
		else if (interactionT == InteractionType::nf) {
			Interaction::absorption(n);
			if (add == true) { // absorption in addedneutron
				atomicAdd(&(Bank->addedNeutronSize), -1);
			}
			else {
				atomicAdd(&(Bank->neutronSize), -1);
			}
		}

	}

	D static void reflection(Neutron& n, C5G7Geometry* Core, double DTS, double DTC, vec3 updatedSurfacePos, double eps) {

		vec3 reflectNormal = { 0.0, 0.0, 0.0 };
		//vec3 afterDTCPos = n.pos + n.dirVec * DTC;
		eps *= 10000;
		if (updatedSurfacePos.x <= eps) {
			reflectNormal = { 1.0, 0.0, 0.0 };
		}
		else if (updatedSurfacePos.y <= eps) {
			reflectNormal = { 0.0, 1.0, 0.0 };
		}
		else if (updatedSurfacePos.z <= eps) {
			reflectNormal = { 0.0, 0.0, 1.0 };
		}
		else if (updatedSurfacePos.x >= Core->x - eps) {
			reflectNormal = { -1.0, 0.0, 0.0 };
		}
		else if (updatedSurfacePos.y >= Core->y - eps) {
			reflectNormal = { 0.0, -1.0, 0.0 };
		}
		else if (updatedSurfacePos.z >= Core->z - eps) {
			reflectNormal = { 0.0, 0.0, -1.0 };
		}
		else {
			// this is fucked
			vec3 updatedCollisionPos = n.pos + n.dirVec * DTC;
			//printf("Pos, (%f, %f, %f),  dir: (%f, %f, %f), DTC: %f\n", n.pos.x, n.pos.y, n.pos.z, n.dirVec.x, n.dirVec.y, n.dirVec.z, DTC);
			printf("Error - reflectnormal not set. Pos, dirvec: (%f,%f,%f), (%f,%f,%f)\nupdatedSurfacePos: (%f,%f,%f), updatedCollisionPos: (%f,%f,%f)\n", n.pos.x, n.pos.y, n.pos.z, n.dirVec.x, n.dirVec.y, n.dirVec.z, updatedSurfacePos.x, updatedSurfacePos.y, updatedSurfacePos.z, updatedCollisionPos.x, updatedCollisionPos.y, updatedCollisionPos.z);
			n.Nullify();
			return;
		}

		vec3 collisionPos = n.pos + n.dirVec * DTS * (1.0 - eps * 10);
		vec3 reflectVec = n.dirVec - reflectNormal * 2 * n.dirVec.dot(reflectNormal);
		//vec3 collisionPos = n.pos + reflectVec * eps * 10000;
		//n.printInfo_Kernel(0);

		n.pos = collisionPos;
		n.dirVec = reflectVec;
		//n.printInfo_Kernel(0);
	}


	H static void reaction_CPU(int idx, Neutron& n, NeutronBank* Bank, XSLibrary* XSLib, Pincell currentPincell, vec3 localPos, GnuAMCM& RNG, double* k_mult, bool passFlag, bool add, double& absorption, double& fission) {
		MatType currentMat = currentPincell.meatOrMod(localPos);
		double outEnergy = 0.0;
		InteractionType interactionT = XSManager::returnInteracitonType(XSLib, currentMat, RNG, n.energy, outEnergy);
		if (interactionT == InteractionType::nel) {
#ifdef INTERACTIONDEBUG 
			printf("idx %d neutron pos (%f,%f,%f) - n,el reaction, scattered from %1.0f to %1.0f\n", idx, n.pos.x, n.pos.y, n.pos.z, n.energy, outEnergy);
#endif
			Interaction::scatter(n, outEnergy, RNG);
		}
		else if (interactionT == InteractionType::ng) {
#ifdef INTERACTIONDEBUG 
			printf("idx %d neutron pod (%f,%f,%f) - n,g reaction\n", idx, n.pos.x, n.pos.y, n.pos.z);
#endif
			Interaction::absorption(n);
			absorption++;
			if (add == true) { // absorption in addedneutron
				Bank->addedNeutronSize -= 1;
			}
			else {
				Bank->neutronSize -= 1;
			}
		}
		else if (interactionT == InteractionType::nf) {
			// fission - function is written inside here (inlined)
			double nu = XSLib->returnMatXSByType(currentPincell.meatOrMod(localPos)).nu[static_cast<int>(n.energy) - 1];
			int fissionNum = static_cast<int>(nu / *k_mult + RNG.uniform(0.0, 1.0));
			
			n.dirVec = vec3::randomUnit(RNG);
			int fissionE = XSManager::returnFissionNeutronEnergy(XSLib->returnMatXSByType(currentPincell.meatOrMod(localPos)), RNG);
			n.energy = fissionE;
			//n.updateWithLength(1.0e-10);
			int addIndex = Bank->addedNeutronIndex;
			Bank->addedNeutronIndex += fissionNum-1;
			Bank->addedNeutronSize += fissionNum-1;
#ifdef FISSIONPRINT 
			printf("Fission on %d, pos: (%f,%f,%f), mat: %s, fission N num: %d, added to index %d.\n", idx, n.pos.x, n.pos.y, n.pos.z, to_string(currentPincell.meatOrMod(localPos)), fissionNum, addIndex);
#endif
			fission += fissionNum;
			for (int i = 0; i < fissionNum - 1; i++) {
				//Bank->addedNeutrons[addIndex + i].status = true;
				int fissionE = XSManager::returnFissionNeutronEnergy(XSLib->returnMatXSByType(currentPincell.meatOrMod(localPos)), RNG);
				Bank->addedNeutrons[addIndex + i].reInitialize(n.pos, vec3::randomUnit(RNG), fissionE, 1.0, passFlag);
			}
		}
	}

	H static void reflection_CPU(Neutron& n, double DTS, vec3 updatedSurfacePos, double eps) {
		vec3 reflectNormal = { 0.0, 0.0, 0.0 };
		if (updatedSurfacePos.x <= eps) {
			reflectNormal = { 1.0, 0.0, 0.0 };
		}
		else if (updatedSurfacePos.y <= eps) {
			reflectNormal = { 0.0, 1.0, 0.0 };
		}
		else if (updatedSurfacePos.z <= eps) {
			reflectNormal = { 0.0, 0.0, 1.0 };
		}
		else {
			// this is fucked
			printf("Error - reflectnormal not set\n");
			n.Nullify();
			return;
		}

		//vec3 collisionPos = n.pos + n.dirVec * DTS;
		vec3 collisionPos = updatedSurfacePos;
		vec3 reflectVec = n.dirVec - reflectNormal * (2 * n.dirVec.dot(reflectNormal));
		collisionPos = collisionPos + reflectVec * eps * 1000;

		n.pos = collisionPos;
		n.dirVec = reflectVec;

	}
};
