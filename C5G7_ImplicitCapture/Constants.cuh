#pragma once
#include <cmath>


namespace Constants {
	// parameters : use gram -> 
	// for mass, variables are declared in amu - convert to g with the amuToG constnat.

	static constexpr double amuToGram = 1.660538921e-24;   // 1 amu = 1.66~~e-24 g
	static constexpr double amuToKilogram = 1.660538921e-27;	// 1 amu = 1.66~~e-27 kg
	static constexpr double M_U235 = 235.0439231;			// based on amu - more like gram/mol
	static constexpr double M_U238 = 238.0507826;			// g/mol
	static constexpr double M_O16 = 15.9949146;			// g/mol
	static constexpr double M_H1 = 1.0078250;			// g/mol
	static constexpr double M_C12 = 12.0;					// g/mol
	static constexpr double M_Neutron = 1.008664916;	// g/mol

	static constexpr double Rho_UO2 = 10.97;				// g/cm^3

	static constexpr double N_A = 6.02214076e+23;		// Avogadro's Number

	static constexpr double PI = 3.141592653589793238462643383279502884197;	//40 fucking digits man it's accurate AF
	//static constexpr double PI_Exact = 
	static constexpr double ElectronC = 1.60217633e-19;
	static constexpr double EulerNum = 2.718281828459045235360287471352;


}

