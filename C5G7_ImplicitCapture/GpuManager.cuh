#pragma once

#include "cudaHeader.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"
#include "Core.cuh"
#include "Tally.cuh"
#include <vector>
#include <algorithm>

struct NeutronBankView {
    Neutron* neutrons;
    Neutron* addedNeutrons;
    int neutronSize;
    int allocatableNeutronNum;
    int addedNeutronSize;
    int addedNeutronIndex;
    unsigned long long seedNo;
};

class GPU_Manager {
    //GPU_Manager() = default;
public:

    // double pointer needed - or put MatXS*& d_ptr. I kept the double pointer so that it stays safe with CUDA style codes. 
    H static void XSLibDeviceAllocator(XSLibrary** d_ptr, XSLibrary& h_instance) {
        cudaMalloc(d_ptr, sizeof(h_instance));
        cudaMemcpy(*d_ptr, &h_instance, sizeof(h_instance), cudaMemcpyHostToDevice);
    }
    // More: if the d_ptr is passed by MatXS* d_ptr, it is passed as value - it doesn't actually change the value(the address) of d_ptr.
    // thus, you need to pass it as the pointer to pointer (MatXS**), or the reference to pointer (MatXS*&)


    H static C5G7Geometry* CoreDeviceAllocator(C5G7Geometry& h_Core, Assembly*& d_bufferAssembly, std::vector<Pincell*>& d_bufferPincellVec) {
        // just to make sure these are clear
        d_bufferAssembly = nullptr;
        d_bufferPincellVec.clear();

        C5G7Geometry* d_Core = nullptr;
        int coreAssemblyNo = h_Core.assemblyNo;

        cudaMalloc(&d_Core, sizeof(C5G7Geometry));
        cudaMalloc(&d_bufferAssembly, sizeof(Assembly) * coreAssemblyNo);
        d_bufferPincellVec.assign(coreAssemblyNo, nullptr);
        std::vector<Assembly> tmp_Assembly(h_Core.assemblyNo);

        for (int i = 0; i < d_bufferPincellVec.size(); i++) {
            int n = h_Core.assembly[i].totalPincellNo();
            cudaMalloc(&d_bufferPincellVec[i], sizeof(Pincell) * h_Core.assembly[i].totalPincellNo());
            cudaMemcpy(d_bufferPincellVec[i], h_Core.assembly[i].pinCells, sizeof(Pincell) * n, cudaMemcpyHostToDevice);
            tmp_Assembly[i] = h_Core.assembly[i];
            tmp_Assembly[i].pinCells = d_bufferPincellVec[i];
        }

        cudaMemcpy(d_bufferAssembly, tmp_Assembly.data(), sizeof(Assembly) * h_Core.assemblyNo, cudaMemcpyHostToDevice);

        C5G7Geometry tmp_Core = h_Core;
        tmp_Core.assembly = d_bufferAssembly;

        cudaMemcpy(d_Core, &tmp_Core, sizeof(C5G7Geometry), cudaMemcpyHostToDevice);

        return d_Core;
    }

    H static TallyC5G7Geometry* CoreTallyDeviceAllocator(TallyC5G7Geometry& h_CoreTally, TallyAssembly*& d_bufferTallyAssembly, std::vector<TallyPincell*>& d_bufferTallyPincellVec) {
        // just to make sure these are clear
        d_bufferTallyAssembly = nullptr;
        d_bufferTallyPincellVec.clear();

        TallyC5G7Geometry* d_CoreTally = nullptr;
        int coreAssemblyNo = h_CoreTally.assemblyNo;

        cudaMalloc(&d_CoreTally, sizeof(TallyC5G7Geometry));
        cudaMalloc(&d_bufferTallyAssembly, sizeof(TallyAssembly) * coreAssemblyNo);
        d_bufferTallyPincellVec.assign(coreAssemblyNo, nullptr);
        std::vector<TallyAssembly> tmp_Assembly(h_CoreTally.assemblyNo);

        for (int i = 0; i < d_bufferTallyPincellVec.size(); i++) {
            int n = h_CoreTally.tallyAssembly[i].totalPincellNo();
            cudaMalloc(&d_bufferTallyPincellVec[i], sizeof(TallyPincell) * h_CoreTally.tallyAssembly[i].totalPincellNo());
            cudaMemcpy(d_bufferTallyPincellVec[i], h_CoreTally.tallyAssembly[i].pinCells, sizeof(TallyPincell) * n, cudaMemcpyHostToDevice);
            tmp_Assembly[i] = h_CoreTally.tallyAssembly[i];
            tmp_Assembly[i].pinCells = d_bufferTallyPincellVec[i];
        }

        cudaMemcpy(d_bufferTallyAssembly, tmp_Assembly.data(), sizeof(TallyAssembly) * h_CoreTally.assemblyNo, cudaMemcpyHostToDevice);

        TallyC5G7Geometry tmp_Core = h_CoreTally;
        tmp_Core.tallyAssembly = d_bufferTallyAssembly;

        cudaMemcpy(d_CoreTally, &tmp_Core, sizeof(TallyC5G7Geometry), cudaMemcpyHostToDevice);

        return d_CoreTally;
    }

    H static inline void FetchCoreTallyToHost(TallyC5G7Geometry& h_CoreTally, TallyAssembly* d_bufferTallyAssembly, const std::vector<TallyPincell*>& d_bufferTallyPincellVec) {
        const int A = h_CoreTally.assemblyNo;
        cudaDeviceSynchronize();

        if (static_cast<int>(d_bufferTallyPincellVec.size()) != A) {
            throw std::runtime_error("d_bufferTallyPincellVec size mismatch with Host's assembly number!\n");
        }

        std::vector<TallyAssembly> tmpAsm(A);
        cudaMemcpy(tmpAsm.data(), d_bufferTallyAssembly, sizeof(TallyAssembly) * A, cudaMemcpyDeviceToHost);

        for (int i = 0; i < A; ++i) {
            const int n = h_CoreTally.tallyAssembly[i].totalPincellNo();
            cudaMemcpy(h_CoreTally.tallyAssembly[i].pinCells, d_bufferTallyPincellVec[i], sizeof(TallyPincell) * n, cudaMemcpyDeviceToHost);
            h_CoreTally.tallyAssembly[i].OOBPincell = tmpAsm[i].OOBPincell;
        }

    }

	

    H static inline void compact_bank_host(NeutronBank& bank) {
        std::vector<Neutron> tmp;
        tmp.reserve(static_cast<size_t>(bank.neutronSize + bank.addedNeutronSize));

        for (int j = 0; j < bank.allocatableNeutronNum; ++j) {
            if (!bank.neutrons[j].isNullified()) {
                bank.neutrons[j].passFlag = false;
                tmp.push_back(bank.neutrons[j]);
            }
            if (!bank.addedNeutrons[j].isNullified()) {
                bank.addedNeutrons[j].passFlag = false;
                tmp.push_back(bank.addedNeutrons[j]);
            }
            bank.neutrons[j].Nullify();
            bank.addedNeutrons[j].Nullify();
        }

        const int cap = bank.allocatableNeutronNum;
        const int total = static_cast<int>(tmp.size());
        const int n0 = std::min(total, cap);
        const int n1 = total - n0;

        for (int j = 0; j < n0; ++j) bank.neutrons[j] = tmp[j];
        for (int j = 0; j < n1; ++j) bank.addedNeutrons[j] = tmp[n0 + j];

        bank.neutronSize = n0;
        bank.addedNeutronSize = n1;
        bank.addedNeutronIndex = n1;
    }

    static inline void compact_bank_device(NeutronBank* d_Bank) {
        NeutronBankView dv;
        cudaMemcpy(&dv, d_Bank, sizeof(NeutronBankView), cudaMemcpyDeviceToHost);

        const int cap = dv.allocatableNeutronNum;

        std::vector<Neutron> hN(static_cast<size_t>(cap));
        std::vector<Neutron> hA(static_cast<size_t>(cap));

        cudaMemcpy(hN.data(), dv.neutrons, sizeof(Neutron) * cap, cudaMemcpyDeviceToHost);
        cudaMemcpy(hA.data(), dv.addedNeutrons, sizeof(Neutron) * cap, cudaMemcpyDeviceToHost);

        std::vector<Neutron> tmp;
        tmp.reserve(static_cast<size_t>(dv.neutronSize + dv.addedNeutronSize));

        for (int j = 0; j < cap; ++j) {
            /*
            if (!hN[j].isNullified() && hN[j].pos.x > 0.0 && hN[j].pos.y > 0.0 && hN[j].pos.z > 0.0) { hN[j].passFlag = false; tmp.push_back(hN[j]); }
            if (!hA[j].isNullified() && hA[j].pos.x > 0.0 && hA[j].pos.y > 0.0 && hA[j].pos.z > 0.0) { hA[j].passFlag = false; tmp.push_back(hA[j]); }
            */

            if (!hN[j].isNullified()) {
                hN[j].passFlag = false;
                tmp.push_back(hN[j]);
            }

            if (!hA[j].isNullified()) {
                hA[j].passFlag = false;
                tmp.push_back(hA[j]);
            }

            hN[j].Nullify();
            hA[j].Nullify();
            

        }

        const int total = static_cast<int>(tmp.size());
        const int n0 = std::min(total, cap);
        const int n1 = total - n0;

        for (int j = 0; j < n0; ++j) hN[j] = tmp[j];
        for (int j = 0; j < n1; ++j) hA[j] = tmp[n0 + j];

        cudaMemcpy(dv.neutrons, hN.data(), sizeof(Neutron) * cap, cudaMemcpyHostToDevice);
        cudaMemcpy(dv.addedNeutrons, hA.data(), sizeof(Neutron) * cap, cudaMemcpyHostToDevice);

        dv.neutronSize = n0;
        dv.addedNeutronSize = n1;
        dv.addedNeutronIndex = n1;

        cudaMemcpy(d_Bank, &dv, sizeof(NeutronBankView), cudaMemcpyHostToDevice);
    }

};
