#pragma once
#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "GpuManager.cuh"
#include "CoreManager.cuh"
#include "XSManager.cuh"
#include "Interaction.cuh"
#include "NeutronBankManager.cuh"
#include "Tally.cuh"
#include "Debug.cuh"


G void fetchNeutronNumber(NeutronBank* Bank, double* d_NeutronNum, double* d_NeutronWeightSum);
G void globalNeutronWeightOffset(NeutronBank* Bank, double* weightOffset);

G void cycle_Neutron(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, int iterTimes, double* fissionCount);
G void addedNeutronPassResetter(NeutronBank* Neutrons);
G void cycle_addedNeutron(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, int iterTimes, double* fissionCount);


H void cycle_Neutron_CPU(NeutronBank* Bank, C5G7Geometry* Core, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, double& absorption, double& fission, double& leak);
H void addedNeutronPassResetter_CPU(NeutronBank* Bank);
H void cycle_addedNeutron_CPU(NeutronBank* Bank, C5G7Geometry* Core, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, double& absorption, double& fission, double& leak);