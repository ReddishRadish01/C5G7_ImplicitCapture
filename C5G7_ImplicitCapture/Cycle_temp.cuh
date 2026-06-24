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

//#define NEUTRONIDXDEBUG
//#define REFLECTIONDEBUG
#define OUTBOUNDDEBUG

constexpr double globaleps = 1.0e-12;

G void cycle_Neutron(NeutronBank* Bank, C5G7Geometry* Core, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= Bank->allocatableNeutronNum) {
        return;
    }
    GnuAMCM RNG(seedNo[idx]);
    
    double eps = 1.0e-10;

    if (!Bank->neutrons[idx].isNullified()) {
        for (int i = 0; i < 100; i++) {
            //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(localNeutron);
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->neutrons[idx]);
            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->neutrons[idx]);

            double DTC = currentAssembly.DTC(Bank->neutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->neutrons[idx]);

            if (DTC < DTS) {    // reaction
                Bank->neutrons[idx].updateWithLength(DTC);
                vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(Bank->neutrons[idx]);
                Interaction::reaction(Bank->neutrons[idx], Bank, XSLib, currentPincell, flooredNeutronPos, RNG, k_mult, passFlag, false);
                seedNo[idx] = RNG.gen();
                return;
            }
            else {  // do:  boundary check / reflection / position update to DTS and feed it back to main loop
                vec3 updatedPos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTC;
                vec3 updatedSurfacePos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTS;
                // handle vaccum boundary neutrons;
                if (updatedPos.x >= Core->x || updatedPos.y >= Core->y || updatedPos.z >= Core->z) {
                    Bank->neutrons[idx].Nullify();
                    atomicAdd(&(Bank->neutronSize), -1);
                    seedNo[idx] = RNG.gen();
                    return;
                }
                if (updatedSurfacePos.x <= eps || updatedSurfacePos.y <= eps || updatedSurfacePos.z <= eps) {
                    Interaction::reflection(Bank->neutrons[idx], DTS, updatedSurfacePos, eps);
                    // this reflection already bumps the neutron to the surface location, with some room for error
                    if (Bank->neutrons[idx].isNullified()) {
                        // error handling - reflection might return a nullified neutron - thus reduce the size of neutron by 1.
                        atomicAdd(&(Bank->neutronSize), -1);
                        seedNo[idx] = RNG.gen();
                        return;
                    }
                    continue;
                }
                // neutron is inside the 
                Bank->neutrons[idx].updateWithLength(DTS + eps);
            }
            //printf("idx %d neutron on %d loop\n", idx, i);
        }
        //printf("idx %d neutron didn't reacted after 100 loops..?\n", idx);
        return;
    }
    return;

}

G void addedNeutronPassResetter(NeutronBank* Neutrons) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }

    if (!Neutrons->addedNeutrons[idx].isNullified()) {
        Neutrons->addedNeutrons[idx].passFlag = false;
    }
}


G void cycle_addedNeutron(NeutronBank* Bank, C5G7Geometry* Core, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= Bank->allocatableNeutronNum) {
        return;
    }
    GnuAMCM RNG(seedNo[idx]);
    double eps = 1.0e-10;

    if (!Bank->addedNeutrons[idx].isNullified()) {
        if (Bank->addedNeutrons[idx].passFlag) {
            Bank->addedNeutrons[idx].passFlag = false;
            return;
        }
    }

    if (!Bank->addedNeutrons[idx].isNullified()) {
        for (int i = 0; i < 100; i++) {
            //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(localNeutron);
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);

            double DTC = currentAssembly.DTC(Bank->addedNeutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->addedNeutrons[idx]);

            if (DTC < DTS) {    // reaction
                Bank->addedNeutrons[idx].updateWithLength(DTC);
                vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(Bank->addedNeutrons[idx]);
                Interaction::reaction(Bank->addedNeutrons[idx], Bank, XSLib, currentPincell, flooredNeutronPos, RNG, k_mult, passFlag, true);
                seedNo[idx] = RNG.gen();
                return;
            }
            else {  // do:  boundary check / reflection / position update to DTS and feed it back to main loop
                vec3 updatedPos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTC;
                vec3 updatedSurfacePos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTS;
                // handle vaccum boundary neutrons;
                if (updatedPos.x >= Core->x || updatedPos.y >= Core->y || updatedPos.z >= Core->z) {
                    Bank->addedNeutrons[idx].Nullify();
                    atomicAdd(&(Bank->addedNeutronSize), -1);
                    seedNo[idx] = RNG.gen();
                    return;
                }
                if (updatedSurfacePos.x <= eps || updatedSurfacePos.y <= eps || updatedSurfacePos.z <= eps) {
                    Interaction::reflection(Bank->addedNeutrons[idx], DTS, updatedSurfacePos, eps);
                    if (Bank->addedNeutrons[idx].isNullified()) {
                        // error handling - reflection might return a nullified neutron - thus reduce the size of neutron by 1.
                        atomicAdd(&(Bank->addedNeutronSize), -1);
                        seedNo[idx] = RNG.gen();
                        return;
                    }
                    continue;
                }
                // neutron is inside the boundary.
                Bank->addedNeutrons[idx].updateWithLength(DTS + eps);
            }
            //printf("idx %d neutron on %d loop\n", idx, i);
        }
        //printf("idx %d neutron didn't reacted after 100 loops...?\n", idx);
        return;
    }
    seedNo[idx] = RNG.gen();
    return;


}