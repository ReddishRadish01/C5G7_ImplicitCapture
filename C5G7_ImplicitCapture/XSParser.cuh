#pragma once

#include "cudaHeader.cuh"

#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

enum class InteractionType {
	ntot,
	nel,
	nTr,
	nAbs,
	ng,
	nf,

};

// you have to make it enum class - prevent it from being recognized as global namespace - a unscoped enum. 
enum class XSType {
	tot,
	trans,
	abs,
	cap,
	fis,
	nu,
	chi,
	el
};

HD static inline const char* to_string(InteractionType t) {
	switch (t) {
	case InteractionType::ntot:     return "ntot";
	case InteractionType::nel:		return "nel";
	case InteractionType::nTr:		return "nTr";
	case InteractionType::nAbs:		return "nAbs";
	case InteractionType::ng:		return "ng";
	case InteractionType::nf:		return "nf";
	default:						return "nel";
	}
}



enum class MatType {
	UO2,
	MOX4_3,
	MOX7_0,
	MOX8_7,
	GT,
	FC,
	MOD,
	Unknown
};

HD static inline const char* to_string(MatType t) {
	switch (t) {
	case MatType::UO2:    return "UO2";
	case MatType::MOX4_3: return "MOX4_3";
	case MatType::MOX7_0: return "MOX7_0";
	case MatType::MOX8_7: return "MOX8_7";
	case MatType::GT:     return "GT";
	case MatType::FC:     return "FC";
	case MatType::MOD:    return "MOD";
	case MatType::Unknown: return "UNKNWON";
	default:              return "UNKNOWN";
	}
}


H static inline std::vector<double> parse_double_from_line(const std::string& line) {
	std::istringstream iss(line);
	std::vector<double> v;
	double x;
	while (iss >> x) v.push_back(x);
	return v;
}


struct MatXS {
	MatType matType{};
	double totalXS[7]{};
	double transXS[7]{};
	double absXS[7]{};
	double capXS[7]{};
	double fisXS[7]{};
	double nu[7]{};
	double chi[7]{};

	double elsXS[7][7]{};

	// Default constructor
	H MatXS() = default;
	H MatXS(MatType matType) : matType(matType) {}

	H MatXS(std::string fileName, MatType matType)
		: matType(matType)
	{
		std::ifstream C5Data(fileName);
		if (C5Data.fail()) {
			std::cout << "Error! File name " << fileName << " doesn't Exist, or Intentionally omitted\n";
		}
		else { std::cout << "Loading C5Data from " << fileName << " Complete!\n"; }
		
		std::string dummy;
		std::getline(C5Data, dummy);

		for (int g = 0; g < 7; g++) {
			std::string line;
			do {
				if (!std::getline(C5Data, line)) {
					throw std::runtime_error("Unexpected EOF wile reading XS rows");
				}
			} while (line.find_first_not_of(" \t\r\n") == std::string::npos);

			std::vector<double> vals = parse_double_from_line(line);
			if (vals.size() == 7) {
				totalXS[g] = vals[0];
				transXS[g] = vals[1];
				absXS[g] = vals[2];
				capXS[g] = vals[3];
				fisXS[g] = vals[4];
				nu[g] = vals[5];
				chi[g] = vals[6];
			}
			else {
				totalXS[g] = vals[0];
				transXS[g] = vals[1];
				absXS[g] = vals[2];
				capXS[g] = vals[3];
				fisXS[g] = 0;
				nu[g] = 0;
				chi[g] = 0;
			}

		}



		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				C5Data >> elsXS[i][j];

			}
		}
		
		
		// this below is to compensate for 10^{-5} degress of bias in transXS compared when every other XS is added up.
		// we force the transXS to be the sum of other XS.

		/*
		for (int g = 0; g < 7; g++) {
			transXS[g] = capXS[g] + fisXS[g];
			for (int i = 0; i < 7; i++) {
				transXS[g] += elsXS[g][i];
			}
		}
		*/
		// likewise for absorption XS
		for (int g = 0; g < 7; g++) {
			absXS[g] = capXS[g] + fisXS[g];
		}

	
		
	}

	HD double returnXSbyType(XSType xsType, double currentEnergy, double destination = 1.0) const {
		switch (xsType) {
		case XSType::tot:		return this->totalXS[static_cast<int>(currentEnergy)-1];
		case XSType::trans:		return this->transXS[static_cast<int>(currentEnergy)-1];
		case XSType::abs:		return this->absXS[static_cast<int>(currentEnergy)-1];
		case XSType::cap:		return this->capXS[static_cast<int>(currentEnergy)-1];
		case XSType::fis:		return this->fisXS[static_cast<int>(currentEnergy)-1];
		case XSType::nu:		return this->nu[static_cast<int>(currentEnergy)-1];
		case XSType::chi:		return this->chi[static_cast<int>(currentEnergy)-1];
		case XSType::el:		return this->elsXS[static_cast<int>(currentEnergy)-1][static_cast<int>(destination)-1];
		}
	}

	H void g7DeviceAllocator(MatXS*& d_G7);

	

};

HD inline double scatterSum(const MatXS& xs, int g) {
	double s = 0.0;
	for (int gp = 0; gp < 7; ++gp) {
		s += xs.elsXS[g - 1][gp];
	}
	return s;
}

HD inline double collisionXS(const MatXS& xs, int g) {
	return xs.capXS[g - 1] + xs.fisXS[g - 1] + scatterSum(xs, g);
}


class XSLibrary {

public:
	MatXS UO2{ MatType::UO2 };
	MatXS MOX4_3{ MatType::MOX4_3 };
	MatXS MOX7_0{ MatType::MOX7_0 };
	MatXS MOX8_7{ MatType::MOX8_7 };
	MatXS GT{ MatType::GT };
	MatXS FC{ MatType::FC };
	MatXS MOD{ MatType::MOD };

	XSLibrary() = default;


	// reference return needed - see the initialize of the MatXSFactory. It requires a reference return.
	HD MatXS& returnMatXSByType(MatType matType) {
		switch (matType) {
		case MatType::UO2:		return UO2;
		case MatType::MOX4_3:	return MOX4_3;
		case MatType::MOX7_0:	return MOX7_0;
		case MatType::MOX8_7:	return MOX8_7;
		case MatType::GT:		return GT;
		case MatType::FC:		return FC;
		case MatType::MOD:		return MOD;
		default:				return MOD;
		}
	}

};


class MatXSFactory {

public:
	XSLibrary XSLib;

	H static void initialize(XSLibrary& XSLib, std::vector<MatXS>& matXS_vec) {
		for (MatXS& XS : matXS_vec) {
			XSLib.returnMatXSByType(XS.matType) = XS;
		}
	}

	H static void initialize(std::vector<MatXS>& matXS_vec) {
		
	}


};

