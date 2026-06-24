#pragma once

#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "CoreManager.cuh"


// this is for finding XS from corresponding location
class XSManager {
public:
	HD static double returnXSByInteractionType(MatType matType, XSLibrary& XSLib, InteractionType interactionType, double in, double out = 0.0) {
		//in = static_cast<int>(in);
		//out = static_cast<int>(out);
		switch(interactionType) {
		case InteractionType::ntot: return XSLib.returnMatXSByType(matType).returnXSbyType(XSType::trans, in, out);
		//case InteractionType::ntr
		
		}
		
		return 0.0;
	}


	HD static double scatteringEnergy(XSLibrary* XSLib, MatType matType, GnuAMCM& RNG, double inEnergy) {
		MatXS XSforCurrentMat = XSLib->returnMatXSByType(matType);
		int currentE = static_cast<int>(inEnergy);
		double totalElas = 0.0;
		for (int i = 0; i < 7; i++) {
			totalElas += XSforCurrentMat.elsXS[currentE - 1][i];
		}
		
		double elasRNG = RNG.uniform(0.0, totalElas);
		double outE = 0.0;

		double group1 = XSforCurrentMat.elsXS[currentE - 1][0];
		double group2 = group1 + XSforCurrentMat.elsXS[currentE - 1][1];
		double group3 = group2 + XSforCurrentMat.elsXS[currentE - 1][2];
		double group4 = group3 + XSforCurrentMat.elsXS[currentE - 1][3];
		double group5 = group4 + XSforCurrentMat.elsXS[currentE - 1][4];
		double group6 = group5 + XSforCurrentMat.elsXS[currentE - 1][5];
		double group7 = group6 + XSforCurrentMat.elsXS[currentE - 1][6];

		if (elasRNG < group1) { outE = 1.0; }
		else if (elasRNG < group2) { outE = 2.0; }
		else if (elasRNG < group3) { outE = 3.0; }
		else if (elasRNG < group4) { outE = 4.0; }
		else if (elasRNG < group5) { outE = 5.0; }
		else if (elasRNG < group6) { outE = 6.0; }
		else if (elasRNG < group7) { outE = 7.0; }
		else { outE = static_cast<double>(currentE); }
		//return static_cast<double>(outE);
		return outE;
		
	}

	HD static InteractionType returnInteracitonType(XSLibrary* XSLib, MatType matType, GnuAMCM& RNG, double inEnergy, double& outEnergy) {
		MatXS XSforCurrentMat = XSLib->returnMatXSByType(matType);
		int currentE = static_cast<int>(inEnergy);

		double xs = RNG.uniform(0.0, XSforCurrentMat.transXS[currentE - 1]);
		double totalElas = 0.0;
		for (int i = 0; i < 7; i++) {
			totalElas += XSforCurrentMat.elsXS[currentE - 1][i];
		}

		if (matType == MatType::GT || matType == MatType::MOD) {
			double cumCap = XSforCurrentMat.capXS[currentE - 1];
			double cumElas = cumCap + totalElas;
			if (xs < cumCap) {
				return InteractionType::ng;
			}
			else if (xs < cumElas) {
				double elasRNG = RNG.uniform(0.0, totalElas);
				double outE = 0.0;

				double group1 = XSforCurrentMat.elsXS[currentE - 1][0];
				double group2 = group1 + XSforCurrentMat.elsXS[currentE - 1][1];
				double group3 = group2 + XSforCurrentMat.elsXS[currentE - 1][2];
				double group4 = group3 + XSforCurrentMat.elsXS[currentE - 1][3];
				double group5 = group4 + XSforCurrentMat.elsXS[currentE - 1][4];
				double group6 = group5 + XSforCurrentMat.elsXS[currentE - 1][5];
				double group7 = group6 + XSforCurrentMat.elsXS[currentE - 1][6];

				if (elasRNG < group1) { outE = 1; }
				else if (elasRNG < group2) { outE = 2; }
				else if (elasRNG < group3) { outE = 3; }
				else if (elasRNG < group4) { outE = 4; }
				else if (elasRNG < group5) { outE = 5; }
				else if (elasRNG < group6) { outE = 6; }
				else if (elasRNG < group7) { outE = 7; }
				else { outE = currentE; }
				outEnergy = outE;

				return InteractionType::nel;
			}
		}
		else {
			double cumCap = XSforCurrentMat.capXS[currentE - 1];
			double cumFis = cumCap + XSforCurrentMat.fisXS[currentE - 1];
			double cumElas = cumFis + totalElas;
			if (xs < cumCap) {
				return InteractionType::ng;
			}
			else if (xs < cumFis) {
				return InteractionType::nf;
			}
			else if (xs < cumElas) {
				double elasRNG = RNG.uniform(0.0, totalElas);
				double outE = 0.0;

				double group1 = XSforCurrentMat.elsXS[currentE - 1][0];
				double group2 = group1 + XSforCurrentMat.elsXS[currentE - 1][1];
				double group3 = group2 + XSforCurrentMat.elsXS[currentE - 1][2];
				double group4 = group3 + XSforCurrentMat.elsXS[currentE - 1][3];
				double group5 = group4 + XSforCurrentMat.elsXS[currentE - 1][4];
				double group6 = group5 + XSforCurrentMat.elsXS[currentE - 1][5];
				double group7 = group6 + XSforCurrentMat.elsXS[currentE - 1][6];

				if (elasRNG < group1) { outE = 1.0; }
				else if (elasRNG < group2) { outE = 2.0; }
				else if (elasRNG < group3) { outE = 3.0; }
				else if (elasRNG < group4) { outE = 4.0; }
				else if (elasRNG < group5) { outE = 5.0; }
				else if (elasRNG < group6) { outE = 6.0; }
				else if (elasRNG < group7) { outE = 7.0; }
				else { outE = currentE; }
				outEnergy = outE;

				return InteractionType::nel;
			}
		}

		//last resort - 
		outEnergy = currentE;
		return InteractionType::nel;
	} 

	HD static int returnFissionNeutronEnergy(MatXS& fisMatXS, GnuAMCM& RNG) {
		double chiRNG = RNG.uniform(0.0, 1.0);
		double chiSelect = 0.0;
		for (int i = 0; i < 7; i++) {
			chiSelect += fisMatXS.chi[i];
			if (chiRNG < chiSelect) {
				return i+1;
			}
		}
		
		// last resort
		return 1;
	}
};