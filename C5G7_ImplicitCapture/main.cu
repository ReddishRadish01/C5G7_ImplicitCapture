
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
#include "Cycle.cuh"

#include "MCdrawer.h"

#include <iomanip>
#include <ctime>
#include <chrono>
#include <sstream>

//#define CPURUN
#define GPURUN
//#define TALLY

//#define XSRESULTDEBUG

#define TALLYWHOLECYCLE


int main() {
	int num = 500000;
	
	int numCycle = 3301;
	int inactiveCycle = 500;
	int activeCycle = 100;
	numCycle = inactiveCycle + activeCycle + 1;
	int iterLimit = 10000;

	int tallyFetchCycleSpec = 200;

	int threadPerBlock = 32;
	int blockPerDim = (num + threadPerBlock - 1) / threadPerBlock;

	double referenceK = 1.183810;
	double h_multK = 0.0;
	std::cout << "input the initial K muliplication factor:\n";
	//std::cin >> h_multK;
	h_multK = referenceK;

	double* d_multK = nullptr;
	cudaMalloc(&d_multK, sizeof(double));
	cudaMemcpy(d_multK, &h_multK, sizeof(double), cudaMemcpyHostToDevice);

	unsigned long long seedNo = 92235922383;

	GnuAMCM h_RNG(seedNo);
	unsigned long long* h_SeedArr = new unsigned long long[num];
	for (int i = 0; i < num; i++) {
		h_SeedArr[i] = (h_RNG.gen() + i) & (0xFFFFFFFFFFFFULL);
	}
	unsigned long long* d_SeedArr = nullptr;
	cudaMalloc(&d_SeedArr, num * sizeof(unsigned long long));
	cudaMemcpy(d_SeedArr, h_SeedArr, num * sizeof(unsigned long long), cudaMemcpyHostToDevice);



	std::vector<MatXS> XS;
	MatXS h_UO2XS("C5txt/UO2.txt", MatType::UO2);
	MatXS h_MOX4_3("C5txt/Mox4_3.txt", MatType::MOX4_3);
	MatXS h_MOX7_0("C5txt/Mox7_0.txt", MatType::MOX7_0);
	MatXS h_MOX8_7("C5txt/Mox8_7.txt", MatType::MOX8_7);
	MatXS h_FC("C5txt/FC.txt", MatType::FC);
	MatXS h_GT("C5txt/GT.txt", MatType::GT);
	MatXS h_Mod("C5txt/Mod.txt", MatType::MOD);
#ifdef XSRESULTDEBUG
	std::cout << h_UO2XS.transXS[0] << " " << h_Mod.transXS[2] << "\n";
#endif

	XS.reserve(7);
	//XS.push_back(h_UO2XS); // uncomfortable
	//XS.push_back(std::move(h_UO2XS)); //h_UO2XS is deprecated
	XS.emplace_back(h_UO2XS);
	XS.emplace_back(h_MOX4_3);
	XS.emplace_back(h_MOX7_0);
	XS.emplace_back(h_MOX8_7);
	XS.emplace_back(h_FC);
	XS.emplace_back(h_GT);
	XS.emplace_back(h_Mod);

	XSLibrary h_XSLib{};
	MatXSFactory::initialize(h_XSLib, XS);
	XSLibrary* d_XSLib = nullptr;
	GPU_Manager::XSLibDeviceAllocator(&d_XSLib, h_XSLib);


	C5G7Geometry h_Core{};
	C5G7GeometryFactory::Initialize(h_Core, "Geometry/C5G7CoreGeometry.txt", "Geometry/UO2Geometry.txt", "Geometry/MOXGeometry.txt");
	Assembly* d_bufferAssembly = nullptr;
	std::vector<Pincell*> d_bufferPincellVec(h_Core.assemblyNo, nullptr);
	C5G7Geometry* d_Core = GPU_Manager::CoreDeviceAllocator(h_Core, d_bufferAssembly, d_bufferPincellVec);

	double h_NeutronNum = static_cast<double>(num);
	double* d_NeutronNum = nullptr;
	double h_NeutronWeightSum = 1.0 * num;
	double initialWeight = 1.0 * num;
	double* d_NeutronWeightSum = nullptr;
	double h_NeutronWeightModifier = 1.0;
	double* d_NeutronWeightModifier = nullptr;
	cudaMalloc(&(d_NeutronNum), sizeof(double));
	cudaMemcpy(d_NeutronNum, &h_NeutronNum, sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_NeutronWeightSum), sizeof(double));
	cudaMemcpy(d_NeutronWeightSum, &h_NeutronWeightSum, sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&(d_NeutronWeightModifier), sizeof(double));
	cudaMemcpy(d_NeutronWeightModifier, &h_NeutronWeightModifier, sizeof(double), cudaMemcpyHostToDevice);


	TallyC5G7Geometry h_CoreTally(h_Core);
	TallyAssembly* d_bufferTallyAssembly = nullptr;
	std::vector<TallyPincell*> d_bufferTallyPincellVec(h_CoreTally.assemblyNo, nullptr);
	TallyC5G7Geometry* d_CoreTally = GPU_Manager::CoreTallyDeviceAllocator(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);




	Neutron h_Neutron{ {4.0, 4.0, 200.0}, {0.0, 0.0, 0.0}, 1.0, 1.0 };
	Neutron* d_Neutron = nullptr;
	cudaMalloc(&d_Neutron, sizeof(Neutron));
	cudaMemcpy(d_Neutron, &h_Neutron, sizeof(Neutron), cudaMemcpyHostToDevice);

	NeutronBank h_Bank(num, seedNo);


	// fetch K to text file

	std::time_t t = std::time(nullptr);
	std::tm tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << "results/k_history_"
		<< (tm.tm_year + 1900)
		<< std::setw(2) << std::setfill('0') << (tm.tm_mon + 1)
		<< std::setw(2) << std::setfill('0') << tm.tm_mday
		<< "_"
		<< std::setw(2) << std::setfill('0') << tm.tm_hour
		<< std::setw(2) << std::setfill('0') << tm.tm_min
		<< std::setw(2) << std::setfill('0') << tm.tm_sec
		<< ".txt";

	std::ostringstream oss2;
	oss2 << "results/flux_Tally"
		<< (tm.tm_year + 1900)
		<< std::setw(2) << std::setfill('0') << (tm.tm_mon + 1)
		<< std::setw(2) << std::setfill('0') << tm.tm_mday
		<< "_"
		<< std::setw(2) << std::setfill('0') << tm.tm_hour
		<< std::setw(2) << std::setfill('0') << tm.tm_min
		<< std::setw(2) << std::setfill('0') << tm.tm_sec
		<< ".txt";

	std::ofstream klog(oss.str(), std::ios::out);
	klog << std::fixed << std::setprecision(6);

	std::ofstream fluxTallyLog(oss2.str(), std::ios::out);
	fluxTallyLog << std::fixed << std::setprecision(6);



	for (int i = 0; i < h_Bank.neutronSize; i++) {
		vec3 randPos = { h_RNG.uniform_open(0, h_Core.x), h_RNG.uniform_open(0, h_Core.y), h_RNG.uniform_open(0, h_Core.z) };
		vec3 centerRandPos = { h_RNG.uniform_open(0, 20), h_RNG.uniform_open(0, 20) ,h_RNG.uniform_open(0, 20) };
		h_Bank.neutrons[i] = Neutron(randPos, vec3::randomUnit(h_RNG), static_cast<double>(h_RNG.int_dist(1, 7)), 1.0);
		//h_Bank.neutrons[i] = Neutron({ 0.1, 0.1, 0.1 }, { -1.0, 0.0, 0.0 }, static_cast<double>(h_RNG.int_dist(1, 7)), 1.0);
		//h_Bank.neutrons[i] = Neutron({ 0.63, 0.63, 0.63 }, { 1.0, 0.0, 0.0 }, static_cast<double>(h_RNG.int_dist(1, 7)), 1.0);
		//h_Bank.neutrons[i] = Neutron(randPos, vec3::randomUnit(h_RNG), 1, 1.0);
		h_Bank.addedNeutrons[i] = Neutron();
		//h_Bank.addedNeutrons[i].status = false;
	}

	NeutronBank* d_Bank = nullptr;
	cudaMalloc(&d_Bank, sizeof(NeutronBank));
	Neutron* d_bufferNeutrons = nullptr; Neutron* d_bufferAddedNeutrons = nullptr;
	cudaMalloc(&d_bufferNeutrons, h_Bank.allocatableNeutronNum * sizeof(Neutron));
	cudaMemcpy(d_bufferNeutrons, h_Bank.neutrons, h_Bank.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyHostToDevice);
	cudaMalloc(&d_bufferAddedNeutrons, h_Bank.allocatableNeutronNum * sizeof(Neutron));
	cudaMemcpy(d_bufferAddedNeutrons, h_Bank.addedNeutrons, h_Bank.allocatableNeutronNum * sizeof(Neutron), cudaMemcpyHostToDevice);
	NeutronBank tmp_Bank = h_Bank;
	tmp_Bank.neutrons = d_bufferNeutrons;
	tmp_Bank.addedNeutrons = d_bufferAddedNeutrons;
	cudaMemcpy(d_Bank, &tmp_Bank, sizeof(NeutronBank), cudaMemcpyHostToDevice);
	

	Neutron h_testNeutron{ {4.0, 4.0, 200.0}, {0.0, 0.0, 0.0}, 1.0, 1.0 };

	
	//for (int i = 0; i < 10; i++) {	Debug::fuelLayoutDebug(h_Core.assembly[i]);	}
	
	//CPUTest(num, &h_XSLib, &h_Core, &h_Bank, h_SeedArr);
	//GPUTest << <blockPerDim, threadPerBlock >> > (num, d_MatXS, d_Core, d_Bank, d_SeedArr);

	double tempK = h_multK;
	double previousNumNeutron = h_Bank.getTotalNeutronNum();
	double meanK = 0.0;
	double M2 = 0.0;
	int errorCounter = 0;
	int cycleNum = 0;
	int activeCount = 0;


	double h_fissionCount = static_cast<double>(num) / 2.05;
	double* d_fissionCount = nullptr;
	cudaMalloc(&(d_fissionCount), sizeof(double));
	cudaMemcpy(d_fissionCount, &h_fissionCount, sizeof(double), cudaMemcpyHostToDevice);
	double fissionCountBuffer = 0.0;

	auto t_start = std::chrono::steady_clock::now();


	for (int cycle = 0; cycle < numCycle; cycle++) {
		cycleNum++;
		double absorption = 0.0;
		double fission = 0.0;
		double leak = 0.0;
		double currentNumNeutron = h_Bank.getTotalNeutronNum();

#ifdef CPURUN
		cycle_addedNeutron_CPU(&h_Bank, &h_Core, &h_CoreTally, &h_XSLib, h_SeedArr, &h_multK, true, absorption, fission, leak);
		addedNeutronPassResetter_CPU(&h_Bank);
		cycle_Neutron_CPU(&h_Bank, &h_Core, &h_CoreTally, &h_XSLib, h_SeedArr, &h_multK, false, absorption, fission, leak);
#endif
		
#ifdef GPURUN
		//ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
		
		cudaMemset(d_NeutronNum, 0, sizeof(double));
		cudaMemset(d_NeutronWeightSum, 0, sizeof(double));

		h_NeutronWeightModifier =   1.0 * (initialWeight/ h_NeutronWeightSum);
		//cudaMemcpy(d_NeutronWeightModifier, &h_NeutronWeightModifier, sizeof(double), cudaMemcpyHostToDevice);


		globalNeutronWeightOffset<<<blockPerDim, threadPerBlock>>>(d_Bank, d_NeutronWeightModifier);

		h_NeutronNum = 0.0; cudaMemcpy(d_NeutronNum, &h_NeutronNum, sizeof(double), cudaMemcpyHostToDevice);
		h_NeutronWeightSum = 0.0; cudaMemcpy(d_NeutronWeightSum, &h_NeutronWeightSum, sizeof(double), cudaMemcpyHostToDevice);

		fetchNeutronNumber << <blockPerDim, threadPerBlock >> > (d_Bank, d_NeutronNum, d_NeutronWeightSum);

		cudaMemcpy(&h_NeutronNum, d_NeutronNum, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_NeutronWeightSum, d_NeutronWeightSum, sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << " NeutronNum: "  << h_NeutronNum << ", Neutron Weight Sum: " << h_NeutronWeightSum << "\n";

		fissionCountBuffer = h_fissionCount;
		h_fissionCount = 0.0; cudaMemcpy(d_fissionCount, &h_fissionCount, sizeof(double), cudaMemcpyHostToDevice);
		
		cycle_addedNeutron << <blockPerDim, threadPerBlock >> > (d_Bank, d_Core, d_CoreTally, d_XSLib, d_SeedArr, d_multK, true, iterLimit, d_fissionCount);
		//CUDA_KERNEL_CHECK();
		addedNeutronPassResetter<<<blockPerDim, threadPerBlock>>>(d_Bank);
		//CUDA_KERNEL_CHECK();
		cycle_Neutron << <blockPerDim, threadPerBlock >> > (d_Bank, d_Core, d_CoreTally, d_XSLib, d_SeedArr, d_multK, false, iterLimit, d_fissionCount);
		//CUDA_KERNEL_CHECK();
		


		cudaMemcpy(&h_fissionCount, d_fissionCount, sizeof(double), cudaMemcpyDeviceToHost);
		//fissionCountBuffer = h_fissionCount;
		double collisionEstK = h_fissionCount / fissionCountBuffer;
		std::cout << "Previous fission: " << fissionCountBuffer << ", Current Fission Count: " << h_fissionCount << ",\tCollisionEstimator: " << h_fissionCount / fissionCountBuffer << "\n";
		// this just fetches only the number of neutrons
		cudaMemcpy(&h_Bank, d_Bank, sizeof(NeutronBank), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_multK, d_multK, sizeof(double), cudaMemcpyDeviceToHost);
		

#endif
		
		currentNumNeutron = h_Bank.getTotalNeutronNum();
		//currentNumNeutron = h_NeutronNum;
		double oldK = h_multK;
		std::cout << "Cycle " << cycle + 1 << ", currentNum: " << currentNumNeutron;
		//h_multK = h_multK * currentNumNeutron / previousNumNeutron;
		h_multK = h_multK *collisionEstK;
		previousNumNeutron = currentNumNeutron;
		std::cout << "\tk: " << h_multK << "\n";
		//std::cout << "\t n count : " << h_Bank.neutronSize << " addn count : " << h_Bank.addedNeutronSize << " addN addIndex : " << h_Bank.addedNeutronIndex;

		if (oldK == h_multK) {
			errorCounter++;
			if (errorCounter == 10) {
				std::cout << "Error in GPU Memory - early termination\n";
				break;
			}
		}
		else {
			errorCounter = 0;
		}

		cudaMemcpy(d_multK, &h_multK, sizeof(double), cudaMemcpyHostToDevice);

		//h_NeutronNum = 0.0;

		h_NeutronNum = currentNumNeutron;
		cudaMemcpy(d_NeutronNum, &h_NeutronNum, sizeof(double), cudaMemcpyHostToDevice);
		// i guess you can use memset?
		//cudaMemset(d_NeutronNum, 0.0, sizeof(double));



#ifdef TALLYWHOLECYCLE
		if (cycle == inactiveCycle + 1) {
			// marks the start of the tally fetch:
			ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
			// this version will fetch the tally, at the last cycle (sum of the tally)
		}
#endif




#ifdef TALLY
		//GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);


		if (cycle > 180 && cycle < 199) {
			
			if (cycle % 2 == 0) {
				std::cout << "Fetching info from core structure flux tally, cycle: " << cycle + 1 << "\n";
				GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
				DumpCoreTallyToText(h_CoreTally, fluxTallyLog, cycle + 1, 10, h_multK);
			}
			ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
		}


		if (cycle == 1) {
			std::cout << "Fetching info from core structure flux tally, cycle: " << cycle + 1 << "\n";
			GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
			// fetch the tally
			DumpCoreTallyToText(h_CoreTally, fluxTallyLog, cycle+1, 10, h_multK);
		}

		if (cycle % tallyFetchCycleSpec == 0 && cycle != 0) {
			std::cout << "Fetching info from core structure flux tally, cycle: " << cycle << "\n";
			GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
			// fetch the tally
			DumpCoreTallyToText(h_CoreTally, fluxTallyLog, cycle+1, 10, h_multK);
		}

		if (cycle % 20 == 0 && cycle < 199) {
			std::cout << "Fetching info from core structure flux tally, cycle: " << cycle + 1 << "\n";
			GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
			// fetch the tally
			DumpCoreTallyToText(h_CoreTally, fluxTallyLog, cycle + 1, 10, h_multK);
		}
		
		if (cycle % tallyFetchCycleSpec - 1 == 0) {
			ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
		}
		if (cycle == 0) {
			ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
		}

		if ((cycle + 1) % 20 == 0 && cycle + 1 < 199) {
			ResetCoreTallyOnDevice(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
		}

#endif



#ifdef CPURUN
		std::cout << "  capture: " << absorption << " , fission neutron num: " << fission << ", leak: " << leak << "\n";
#endif

		klog << (cycle + 1) << " " << h_multK << "\n";
		if (cycle > inactiveCycle) {
			activeCount += 1;
			double x = h_multK;
			double delta = x - meanK;
			meanK += delta / activeCount;
			double delta2 = x - meanK;
			M2 += delta * delta2;

		}

		
		if (h_Bank.addedNeutronIndex > h_Bank.allocatableNeutronNum * 0.8) {
			std::cout << "Merging:\n";
#ifdef CPURUN
			std::vector<Neutron> NeutronContainer;
			for (int j = 0; j < h_Bank.allocatableNeutronNum; j++) {
				if (!h_Bank.neutrons[j].isNullified()) {
					NeutronContainer.reserve(1);
					h_Bank.neutrons[j].passFlag = false;
					NeutronContainer.emplace_back(h_Bank.neutrons[j]);
					h_Bank.neutrons[j].Nullify();	// flush all the old neutrons
				}

				if (!h_Bank.addedNeutrons[j].isNullified()) {
					NeutronContainer.reserve(1);
					h_Bank.addedNeutrons[j].passFlag = false;
					NeutronContainer.emplace_back(h_Bank.addedNeutrons[j]);
					h_Bank.addedNeutrons[j].Nullify();
				}
			}

			if (h_Bank.allocatableNeutronNum < NeutronContainer.size()) {
				h_Bank.neutronSize = h_Bank.allocatableNeutronNum;
				h_Bank.addedNeutronSize = NeutronContainer.size() - h_Bank.allocatableNeutronNum;
				h_Bank.addedNeutronIndex = h_Bank.addedNeutronSize;
				std::cout << "Neutron vector size: " << NeutronContainer.size() << "\n";

				for (int j = 0; j < h_Bank.allocatableNeutronNum; j++) {
					h_Bank.neutrons[j] = NeutronContainer[j];
				}
				for (int j = 0; j < h_Bank.addedNeutronSize; j++) {
					h_Bank.addedNeutrons[j] = NeutronContainer[h_Bank.neutronSize + j];
				}

				std::cout << "After sorting: Neutron size: " << h_Bank.getTotalNeutronNum() << "\n";
			}
			else {
				h_Bank.neutronSize = NeutronContainer.size();
				h_Bank.addedNeutronIndex = 0;
				h_Bank.addedNeutronSize = 0;
				std::cout << "Neutron vector size: " << NeutronContainer.size() << "\n";

				for (int j = 0; j < h_Bank.neutronSize; j++) {
					h_Bank.neutrons[j] = NeutronContainer[j];
				}
				std::cout << "After sorting: Neutron size: " << h_Bank.getTotalNeutronNum() << "\n";
			}
		}
#endif

#ifdef GPURUN
			GPU_Manager::compact_bank_device(d_Bank);
		}
#endif
		
		std::cout << "End Of Cycle\n";
	}
	auto t_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = t_end - t_start;

	std::cout << "Total Time: " << elapsed.count() << "seconds. Average: " << elapsed.count() / cycleNum << " seconds per cycle.\n";

	if (activeCount >= 2) {
		double var = M2 / (activeCount - 1);
		double stddev = std::sqrt(var);
		double stderr_mean = stddev / std::sqrt(static_cast<double>(activeCount));

		std::cout << "\n\nActive cycles: " << activeCount << "\n";
		std::cout << "k_mean: " << std::fixed << std::setprecision(7) << meanK << "\n";
		std::cout << "k_stddev(cycle): " << stddev << "\n";
		std::cout << "k_stderr(mean): " << stderr_mean << "\n";

		std::cout << "Reactivity(rho): " << (meanK - 1.0) / meanK << ", Per Cent error compared to reference: " <<  (referenceK - 1) / referenceK - (meanK - 1.0) / meanK << "\n";

		klog << "\n\nActive cycles: " << activeCount << "\n";
		klog << "k_mean: " << meanK << "\n";
		klog << "k_stddev(cycle): " << stddev << "\n";
		klog << "k_stderr(mean): " << stderr_mean << "\n";
		klog << "Reactivity(rho): " << (meanK - 1.0) / meanK << ", Per Cent error compared to reference: " << ((referenceK - 1) / referenceK - (meanK - 1.0) / meanK) * 100 << "\n";
		klog << "Total Time: " << elapsed.count() << "seconds. Average: " << elapsed.count() / cycleNum << " seconds per cycle.\n";
	
	}

#ifdef TALLYWHOLECYCLE
	GPU_Manager::FetchCoreTallyToHost(h_CoreTally, d_bufferTallyAssembly, d_bufferTallyPincellVec);
	DumpCoreTallyToText(h_CoreTally, fluxTallyLog, activeCycle + inactiveCycle, 10, meanK);

#endif


	//std::cout << "\n initial Neutron number: " << num << ", for cycle of " << cycleNum - inactiveCycle << ", average k = " << meanK << "\n";

	klog.close();
	fluxTallyLog.close();

	delete[] h_SeedArr;
	h_SeedArr = nullptr;

	if (d_bufferNeutrons) cudaFree(d_bufferNeutrons);
	if (d_bufferAddedNeutrons) cudaFree(d_bufferAddedNeutrons);
	if (d_Bank) cudaFree(d_Bank);

	if (d_Neutron) cudaFree(d_Neutron);

	for (auto& p : d_bufferPincellVec) {
		if (p) cudaFree(p);
		p = nullptr;
	}

	for (auto& p : d_bufferTallyPincellVec) {
		if (p) cudaFree(p);
		p = nullptr;
	}

	if (d_bufferAssembly) cudaFree(d_bufferAssembly);
	if (d_bufferTallyAssembly) cudaFree(d_bufferTallyAssembly);

	if (d_Core) cudaFree(d_Core);
	if (d_CoreTally) cudaFree(d_CoreTally);
	if (d_XSLib) cudaFree(d_XSLib);

	if (d_SeedArr) cudaFree(d_SeedArr);

	if (d_multK) cudaFree(d_multK);

	// Optional: reset device (helps detect leaks in some tools)
	// this would really help
	cudaDeviceReset();
}