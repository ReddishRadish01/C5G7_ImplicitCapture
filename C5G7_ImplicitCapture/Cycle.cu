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

//#define TALLYINCYCLE
#define TALLYFissionSourceDist


constexpr double globaleps = 1.0e-10;

HD inline double sign_nonzero(double v) {
    return (v >= 0.0) ? 1.0 : -1.0;
}

HD inline bool near_multiple(double x, double pitch, double tol) {
    if (pitch <= 0.0) return false;

    double r = fmod(x, pitch);
    if (r < 0.0) r += pitch;

    return (r < tol) || ((pitch - r) < tol);
}

HD inline bool bumpAcrossFlatCellPlane(Neutron& n, const Assembly& a, double tol) {
    bool bumped = false;

    vec3 local = n.pos - a.startPos;

    double dx = a.length.x / a.xNum;
    double dy = a.length.y / a.yNum;
    double dz = a.length.z / a.zNum;

    if (fabs(n.dirVec.x) > 1.0e-14 &&
        local.x > tol && local.x < a.length.x - tol &&
        near_multiple(local.x, dx, tol)) {
        n.pos.x += sign_nonzero(n.dirVec.x) * tol;
        bumped = true;
    }

    if (fabs(n.dirVec.y) > 1.0e-14 &&
        local.y > tol && local.y < a.length.y - tol &&
        near_multiple(local.y, dy, tol)) {
        n.pos.y += sign_nonzero(n.dirVec.y) * tol;
        bumped = true;
    }

    if (fabs(n.dirVec.z) > 1.0e-14 &&
        local.z > tol && local.z < a.length.z - tol &&
        near_multiple(local.z, dz, tol)) {
        n.pos.z += sign_nonzero(n.dirVec.z) * tol;
        bumped = true;
    }

    return bumped;
}


G void globalNeutronWeightOffset(NeutronBank* Bank, double* weightModifier) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= Bank->allocatableNeutronNum) {
		return;
	}
	if (!Bank->neutrons[idx].isNullified()) {
		Bank->neutrons[idx].weight *= *weightModifier;
	}
	if (!Bank->addedNeutrons[idx].isNullified()) {
		Bank->addedNeutrons[idx].weight *= *weightModifier;
	}
}


__inline__ __device__
double warpReduceSum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__
double blockReduceSum(double val) {
    static __shared__ double shared[32]; // max 1024 threads / 32 warps

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    val = 0.0;

    if (threadIdx.x < blockDim.x / warpSize) {
        val = shared[lane];
    }

    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

G void fetchNeutronNumber(
    NeutronBank* Bank,
    double* d_NeutronNum,
    double* d_NeutronWeightSum
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    double localNum = 0.0;
    double localWeight = 0.0;

    if (idx < Bank->allocatableNeutronNum) {
        if (!Bank->neutrons[idx].isNullified()) {
            localNum += 1.0;
            localWeight += Bank->neutrons[idx].weight;
        }

        if (!Bank->addedNeutrons[idx].isNullified()) {
            localNum += 1.0;
            localWeight += Bank->addedNeutrons[idx].weight;
        }
    }

    double blockNum = blockReduceSum(localNum);
    double blockWeight = blockReduceSum(localWeight);

    if (threadIdx.x == 0) {
        atomicAdd(d_NeutronNum, blockNum);
        atomicAdd(d_NeutronWeightSum, blockWeight);
    }
}

/*
// Naive atomicAdd approach - not efficient
G void fetchNeutronNumber(NeutronBank* Bank, double* d_NeutronNum, double* d_NeutronWeightSum) {
    // here maybe we can use parallel reduction?

    //int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Bank->allocatableNeutronNum) {
        return;
    }

    if (!Bank->neutrons[idx].isNullified()) {
        atomicAdd(d_NeutronNum, 1.0);
        atomicAdd(d_NeutronWeightSum, Bank->neutrons[idx].weight);
    }

    if (!Bank->addedNeutrons[idx].isNullified()) {
        atomicAdd(d_NeutronNum, 1.0);
        atomicAdd(d_NeutronWeightSum, Bank->addedNeutrons[idx].weight);
    }
}
*/

G void weightParallelReductionSum(NeutronBank* Bank, double*) {
    
}


G void cycle_Neutron(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, int iterTimes, double* fissionWeight) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= Bank->allocatableNeutronNum) {
        return;
    }
    GnuAMCM RNG(seedNo[idx]);

    double eps = 1.0e-13;
    int reflectionCounter = 0;

    if (!Bank->neutrons[idx].isNullified()) {
        for (int i = 0; i < iterTimes; i++) {


            //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(localNeutron);
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->neutrons[idx]);
            // currently, currentAssembly pushes the neutron inside the assembly by its dirvec direction
            double fissionCount = 0.0;
#ifdef TALLYINCYCLE
            TallyAssembly currentTallyAssembly = CoreTally->returnAssemblyByNeutron(Bank->neutrons[idx]);
#endif
#ifdef TALLYFissionSourceDist
            TallyAssembly currentTallyAssembly = CoreTally->returnAssemblyByNeutron(Bank->neutrons[idx]);
#endif


            if (Bank->neutrons[idx].isNullified()) {
                atomicAdd(&(Bank->neutronSize), -1);
#ifdef OUTBOUNDDEBUG
                printf("Neutron idx %d nullfied because Assembly positioning got fucked\n", idx);
#endif
                seedNo[idx] = RNG.gen();
                return;
            }

            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->neutrons[idx]);
            vec3 flooredNeutronPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->neutrons[idx]);
#ifdef TALLYINCYCLE
            TallyPincell currentTallyPincell = currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]);
#endif
#ifdef TALLYFissionSourceDist
            TallyPincell currentTallyPincell = currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]);
#endif

            // handling OOB pincells returned from Assembly indexing error - OOB pincell returned from function returnPincellByPos()
            if (currentPincell.sideLength < 1e-4 || currentPincell.height < 1e-4) {
                // on-surface exceptions - bump it into other cell, and continue
                // this is error exception control - almost no neutron ends up in this branch.
                /*
                if ((Bank->neutrons[idx].pos.x >= currentAssembly.startPos.x + currentAssembly.length.x - eps * 100) && (Bank->neutrons[idx].pos.x <= currentAssembly.startPos.x + currentAssembly.length.x + eps * 100) ||
                    (Bank->neutrons[idx].pos.y >= currentAssembly.startPos.y + currentAssembly.length.y - eps * 100) && (Bank->neutrons[idx].pos.y <= currentAssembly.startPos.y + currentAssembly.length.y + eps * 100) ||
                    (Bank->neutrons[idx].pos.z >= currentAssembly.startPos.z + currentAssembly.length.z - eps * 100) && (Bank->neutrons[idx].pos.z <= currentAssembly.startPos.z + currentAssembly.length.z + eps * 100)) {
                    Bank->neutrons[idx].updateWithLength(eps * 10000);
                    //printf("lmao you leak here");
                    continue;
                }
                
                printf("error in returning pincell of neutron idx %d, - nullifying this neutron\n", idx);
                Bank->neutrons[idx].Nullify();
                atomicAdd(&(Bank->neutronSize), -1);
                */

				currentAssembly = Core->unstuckNeutron(Bank->neutrons[idx]);
				currentPincell = currentAssembly.returnPincellByPos(Bank->neutrons[idx]);
                flooredNeutronPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->neutrons[idx]);

                /*
                seedNo[idx] = RNG.gen();
                return;
                */
            }


            double DTC = currentAssembly.DTC(Bank->neutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->neutrons[idx]);

            if (DTC < DTS) {    // reaction
                Bank->neutrons[idx].updateWithLength(DTC);
                //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(Bank->neutrons[idx]);

                // implicit reaction - in order to accomodate neutron weight and some RR shits
                Interaction::reaction_implicit(Bank->neutrons[idx], Bank, XSLib, currentPincell, flooredNeutronPos, RNG, k_mult, passFlag, false, fissionWeight, fissionCount);

#ifdef TALLYINCYCLE
                if (currentPincell.meatOrMod(flooredNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).modTally), DTC); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).pinTally), DTC); }
#endif

#ifdef TALLYFissionSourceDist
                if (currentPincell.meatOrMod(flooredNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).modTally), fissionCount); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).pinTally), fissionCount); }
#endif

                Interaction::russianRoulette(Bank->neutrons[idx], Bank, RNG, 0.25, 0.5, false);
                if (Bank->neutrons[idx].isNullified()) {
                    seedNo[idx] = RNG.gen();
                    return;
                }

            }
            else {  // do:  boundary check / reflection / position update to DTS and feed it back to main loop
                //vec3 updatedPos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTC;
#ifdef TALLYINCYCLE
                if (currentPincell.meatOrMod(flooredNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).modTally), DTS); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->neutrons[idx]).pinTally), DTS); }
#endif
                vec3 afterDTSPos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTS;
                vec3 afterDTCPos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTC;

                // ** Neutrons leaked through vaccum boundary
                
                if (afterDTSPos.x >= Core->x - eps * 100 || afterDTSPos.y >= Core->y - eps * 100 || afterDTSPos.z >= Core->z - eps * 100) {
                    Bank->neutrons[idx].Nullify();
                    atomicAdd(&(Bank->neutronSize), -1);
                    seedNo[idx] = RNG.gen();
                    return;
                }

                
                // ** Neutrons reflected at boundaries - push neutron to the surface, bump it (with reflection function) and continue the loop
                //if (afterDTSPos.x < eps || afterDTSPos.y < eps || afterDTSPos.z < eps || afterDTSPos.x >= Core->x - eps || afterDTSPos.y >= Core->y - eps || afterDTSPos.z >= Core->z - eps) {
                if (afterDTSPos.x < eps* 100 || afterDTSPos.y < eps * 100 || afterDTSPos.z < eps * 100) {
                    if (true) {
                        //Bank->neutrons[idx].printInfo_Kernel(idx);
                        Interaction::reflection(Bank->neutrons[idx], Core, DTS, DTC, afterDTSPos, eps * 100);
#ifdef REFLECTIONDEBUG
                        reflectionCounter++;
                        if (reflectionCounter > 1) {
                            printf("Neutron reflected twice on ");
                            Bank->neutrons[idx].printInfo_Kernel(idx);
                        }
#endif
                        // this reflection already bumps the neutron to the surface location, with some room for error
                        if (Bank->neutrons[idx].isNullified()) {
                            // error handling - reflection might return a nullified neutron - thus reduce the size of neutron by 1.
                            atomicAdd(&(Bank->neutronSize), -1);
                            seedNo[idx] = RNG.gen();
                            return;
                        }
                        //continue;
                    }
                }
                else {
                    // Pin-moderator trans boundary situation - seems like neutron gets stuck with DTS of zero - thus even with eps push at the line above, it still gets stuck.
                    if (DTS < eps * 10) {
                        const double boundaryBump = 1.0e-8;

                        bool bumped = bumpAcrossFlatCellPlane(Bank->neutrons[idx], currentAssembly, boundaryBump);

                        if (!bumped) {
                            // fallback for pin/mod cylindrical boundary or any unresolved near-surface case
                            Bank->neutrons[idx].updateWithLength(boundaryBump);
                        }

                        continue;
                                              
                        /*
                        double localX = fmod(afterDTSPos.x, 1.26);
                        double localY = fmod(afterDTSPos.y, 1.26);
                        double distSq = (localX - 0.63) * (localX - 0.63) + (localY - 0.63) * (localY - 0.63);
                        double radiusSq = 0.54 * 0.54;
                        if (distSq < radiusSq + eps * 100 && distSq > radiusSq - eps * 100) {
                            //printf("Pushed neutron at the rod-moderator boundary, because DTS was so small\n");
                            //Bank->neutrons[idx].printInfo_Kernel(idx);
                            Bank->neutrons[idx].updateWithLength(eps * 1.0e+4);
                            //Bank->neutrons[idx].printInfo_Kernel(idx);
                        }
                        */
                    }
                    else {
                        // neutron is inside boundary, no reflection, and entered different cell(Trans-surface)
                        Bank->neutrons[idx].updateWithLength(DTS + eps * 100);
                    }

                }
                
                
            }
            //printf("idx %d neutron on %d loop\n", idx, i);

            if (i == iterTimes * 7 / 10 && DTS < eps * 10) {
                printf("idx %d neutron on %d loop struggling: (%f,%f,%f), (%f,%f,%f), DTC:%e, DTS:%e\n", idx, i, Bank->neutrons[idx].pos.x, Bank->neutrons[idx].pos.y, Bank->neutrons[idx].pos.z, Bank->neutrons[idx].dirVec.x, Bank->neutrons[idx].dirVec.y, Bank->neutrons[idx].dirVec.z, DTC, DTS);
                Bank->neutrons[idx].updateWithLength(eps * 1.0e+3);
            }

            if (i == iterTimes - 1) {
                //Bank->neutrons[idx].printInfo_Kernel(idx);
                //printf("idx %d neutron (%f,%f,%f) didn't reacted after %d loops..? weight: %f\n", idx, Bank->neutrons[idx].pos.x, Bank->neutrons[idx].pos.y, Bank->neutrons[idx].pos.z, iterTimes, Bank->neutrons[idx].weight);
                //Bank->neutrons[idx].updateWithLength(1.0e-7);

                printf("STUCK neutron idx %d: pos=(%.17g,%.17g,%.17g), dir=(%.17g,%.17g,%.17g), weight=%.17g\n",
                    idx,
                    Bank->neutrons[idx].pos.x, Bank->neutrons[idx].pos.y, Bank->neutrons[idx].pos.z,
                    Bank->neutrons[idx].dirVec.x, Bank->neutrons[idx].dirVec.y, Bank->neutrons[idx].dirVec.z,
                    Bank->neutrons[idx].weight);

                Bank->neutrons[idx].updateWithLength(1.0e-8);
                seedNo[idx] = RNG.gen();
                return;
                
                //vec3 bump = { 1.0e-3, 1.0e-3, 1.0e-3 };
                //vec3 randPos = { RNG.uniform(0, Core->x), RNG.uniform(0, Core->y), RNG.uniform(0, Core->z) };
                //Bank->neutrons[idx].pos = Bank->neutrons[idx].pos - bump;
                //Bank->neutrons[idx].pos = randPos;
                //Bank->neutrons[idx].dirVec = vec3::randomUnit(RNG);
                //seedNo[idx] = RNG.gen();
                //Bank->neutrons[idx].Nullify();
                //atomicAdd(&(Bank->neutronSize), -1);
                //return;
            }
        }


        //printf("idx %d neutron didn't reacted after %d loops..?\n", idx, iterTimes);
        seedNo[idx] = RNG.gen();
        return;

    }
    seedNo[idx] = RNG.gen();
    return;

}

G void addedNeutronPassResetter(NeutronBank* Neutrons) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Neutrons->allocatableNeutronNum) { return; }

    if (!Neutrons->addedNeutrons[idx].isNullified()) {
        Neutrons->addedNeutrons[idx].passFlag = false;
    }
}


G void cycle_addedNeutron(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, int iterTimes, double* fissionWeight) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= Bank->allocatableNeutronNum) {
        return;
    }

    GnuAMCM RNG(seedNo[idx]);
    double eps = 1.0e-13;
    int reflectionCounter = 0;

    if (!Bank->addedNeutrons[idx].isNullified()) {
        if (Bank->addedNeutrons[idx].passFlag) {
            Bank->addedNeutrons[idx].passFlag = false;
            return;
        }
    }

    if (!Bank->addedNeutrons[idx].isNullified()) {
        for (int i = 0; i < iterTimes; i++) {


            //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(localNeutron);
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
            double fissionCount = 0.0;
#ifdef TALLYINCYCLE
            TallyAssembly currentTallyAssembly = CoreTally->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
#endif
#ifdef TALLYFissionSourceDist
            TallyAssembly currentTallyAssembly = CoreTally->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
#endif

            if (Bank->addedNeutrons[idx].isNullified()) {
                atomicAdd(&(Bank->addedNeutronSize), -1);
#ifdef OUTBOUNDDEBUG
                printf("Neutron idx %d nullfied because Assembly positioning got fucked\n", idx);
#endif
                seedNo[idx] = RNG.gen();
                return;
            }

            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);
            vec3 flooredAddedNeutronPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->addedNeutrons[idx]);
                
#ifdef TALLYINCYCLE
            TallyPincell currentTallyPincell = currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);
#endif
#ifdef TALLYFissionSourceDist
            TallyPincell currentTallyPincell = currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);
#endif

            // OOB Pincell handling
            if (currentPincell.sideLength < 1e-4 || currentPincell.height < 1e-4) {
                /*
                if ((Bank->addedNeutrons[idx].pos.x >= currentAssembly.startPos.x + currentAssembly.length.x - eps * 100) && (Bank->addedNeutrons[idx].pos.x <= currentAssembly.startPos.x + currentAssembly.length.x + eps * 100) ||
                    (Bank->addedNeutrons[idx].pos.y >= currentAssembly.startPos.y + currentAssembly.length.y - eps * 100) && (Bank->addedNeutrons[idx].pos.y <= currentAssembly.startPos.y + currentAssembly.length.y + eps * 100) ||
                    (Bank->addedNeutrons[idx].pos.z >= currentAssembly.startPos.z + currentAssembly.length.z - eps * 100) && (Bank->addedNeutrons[idx].pos.z <= currentAssembly.startPos.z + currentAssembly.length.z + eps * 100)) {
                    Bank->addedNeutrons[idx].updateWithLength(eps * 10000);
                    continue;
                    printf("Lmao you leak here");
                }



                printf("error in returning pincell of adddedNeutron idx %d, - nullifying this neutron\n", idx);
                Bank->addedNeutrons[idx].Nullify();
                atomicAdd(&(Bank->addedNeutronSize), -1);
                seedNo[idx] = RNG.gen();
                return;
                */
                currentAssembly = Core->unstuckNeutron(Bank->addedNeutrons[idx]);
                currentPincell = currentAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);
                flooredAddedNeutronPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->addedNeutrons[idx]);
            }

            double DTC = currentAssembly.DTC(Bank->addedNeutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->addedNeutrons[idx]);

            if (DTC < DTS) {    // reaction
                Bank->addedNeutrons[idx].updateWithLength(DTC);
                //vec3 flooredNeutronPos = Core->assembly->returnFlooredNeutronPosInPincell(Bank->addedNeutrons[idx]);

                Interaction::reaction_implicit(Bank->addedNeutrons[idx], Bank, XSLib, currentPincell, flooredAddedNeutronPos, RNG, k_mult, passFlag, true, fissionWeight, fissionCount);

#ifdef TALLYINCYCLE
                if (currentPincell.meatOrMod(flooredAddedNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).modTally), DTC); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).pinTally), DTC); }
#endif
#ifdef TALLYFissionSourceDist
                if (currentPincell.meatOrMod(flooredAddedNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).modTally), fissionCount); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).pinTally), fissionCount); }
#endif

                Interaction::russianRoulette(Bank->addedNeutrons[idx], Bank, RNG, 0.25, 0.5, true);
                if (Bank->addedNeutrons[idx].isNullified()) {
                    seedNo[idx] = RNG.gen();
                    return;
                }
            }
            else {  // do:  boundary check / reflection / position update to DTS and feed it back to main loop
                //vec3 updatedPos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTC;
#ifdef TALLYINCYCLE
                if (currentPincell.meatOrMod(flooredAddedNeutronPos) == MatType::MOD) { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).modTally), DTS); }
                else { atomicAdd(&(currentTallyAssembly.returnPincellByPos(Bank->addedNeutrons[idx]).pinTally), DTS); }
#endif
                vec3 afterDTSPos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTS;
                vec3 afterDTCPos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTC;

                // handle vaccum boundary neutrons;
                //if (afterDTSPos.x >= Core->x || afterDTSPos.y >= Core->y || afterDTSPos.z >= Core->z) {
                
                if (afterDTSPos.x >= Core->x - eps * 100 || afterDTSPos.y >= Core->y - eps * 100 || afterDTSPos.z >= Core->z - eps * 100) {
                    Bank->addedNeutrons[idx].Nullify();
                    atomicAdd(&(Bank->addedNeutronSize), -1);
                    seedNo[idx] = RNG.gen();
                    return;
                }
                

                // Reflection
                //if (afterDTSPos.x < eps || afterDTSPos.y < eps || afterDTSPos.z < eps || afterDTSPos.x >= Core->x - eps || afterDTSPos.y >= Core->y - eps || afterDTSPos.z >= Core->z - eps) {
                if (afterDTSPos.x < eps * 100 || afterDTSPos.y < eps * 100 || afterDTSPos.z < eps * 100) {
                    if (true) {
						//Bank->addedNeutrons[idx].printInfo_Kernel(idx);
                        Interaction::reflection(Bank->addedNeutrons[idx], Core, DTS, DTC, afterDTSPos, eps * 100);
#ifdef REFLECTIONDEBUG
                        if (reflectionCounter > 1) {
                            printf("Neutron reflected twice on ");
                            Bank->addedNeutrons[idx].printInfo_Kernel(idx);
                        }
#endif
                        if (Bank->addedNeutrons[idx].isNullified()) {
                            // error handling - reflection might return a nullified neutron - thus reduce the size of neutron by 1.
                            atomicAdd(&(Bank->addedNeutronSize), -1);
                            seedNo[idx] = RNG.gen();
                            return;
                        }
                    }
                }
                else {
                    // Pin-moderator trans boundary situation - seems like neutron gets stuck with DTS of zero - thus even with eps push at the line above, it still gets stuck.
                    if (DTS < eps * 10) {
                        /*
						double localX = fmod(afterDTSPos.x, 1.26);
                        double localY = fmod(afterDTSPos.y, 1.26);
						double distSq = (localX - 0.63) * (localX - 0.63) + (localY - 0.63) * (localY - 0.63);
                        double radiusSq = 0.54 * 0.54;
                        if (distSq < radiusSq + eps * 100 && distSq > radiusSq - eps * 100) {
                            printf("Pushed neutron at the rod-moderator boundary, because DTS was so small\n");
                            //Bank->addedNeutrons[idx].printInfo_Kernel(idx);
                            Bank->addedNeutrons[idx].updateWithLength(eps * 1.0e+4);
                            //Bank->addedNeutrons[idx].printInfo_Kernel(idx);
                        }
                        */
                        const double boundaryBump = 1.0e-8;

                        bool bumped = bumpAcrossFlatCellPlane(Bank->addedNeutrons[idx], currentAssembly, boundaryBump);

                        if (!bumped) {
                            // fallback for pin/mod cylindrical boundary or any unresolved near-surface case
                            Bank->addedNeutrons[idx].updateWithLength(boundaryBump);
                        }

                        continue;
                    }
                    else {
                        // neutron is inside boundary, no reflection, and entered different cell(Trans-surface)
                        Bank->addedNeutrons[idx].updateWithLength(DTS + eps * 100);
                    }
                }
                
                
            }
            //printf("idx %d neutron on %d loop\n", idx, i);


            if (i == iterTimes * 6 / 10 && DTS < eps * 10) {
                printf("idx %d neutron on %d loop struggling: (%f,%f,%f), (%f,%f,%f), DTC:%e, DTS:%e\n", idx, i, Bank->addedNeutrons[idx].pos.x, Bank->addedNeutrons[idx].pos.y, Bank->addedNeutrons[idx].pos.z, Bank->addedNeutrons[idx].dirVec.x, Bank->addedNeutrons[idx].dirVec.y, Bank->addedNeutrons[idx].dirVec.z, DTC, DTS);
                Bank->addedNeutrons[idx].updateWithLength(eps * 1.0e+4);
            }

            if (i == iterTimes - 1) {
                //Bank->addedNeutrons[idx].printInfo_Kernel(idx);
                //printf("idx %d addedNeutron (%f,%f,%f) didn't reacted after %d loops..? weight: %f\n", idx, Bank->addedNeutrons[idx].pos.x, Bank->addedNeutrons[idx].pos.y, Bank->addedNeutrons[idx].pos.z, iterTimes, Bank->addedNeutrons[idx].weight);
                //Bank->addedNeutrons[idx].updateWithLength(1.0e-4);

                printf("STUCK neutron idx %d: pos=(%.17g,%.17g,%.17g), dir=(%.17g,%.17g,%.17g), weight=%.17g\n",
                    idx,
                    Bank->addedNeutrons[idx].pos.x, Bank->addedNeutrons[idx].pos.y, Bank->addedNeutrons[idx].pos.z,
                    Bank->addedNeutrons[idx].dirVec.x, Bank->addedNeutrons[idx].dirVec.y, Bank->addedNeutrons[idx].dirVec.z,
                    Bank->addedNeutrons[idx].weight);

                Bank->addedNeutrons[idx].updateWithLength(1.0e-8);
                seedNo[idx] = RNG.gen();
                return;
                
                //vec3 bump = { 1.0e-3, 1.0e-3, 1.0e-3 };
                //vec3 randPos = { RNG.uniform(0, Core->x), RNG.uniform(0, Core->y), RNG.uniform(0, Core->z) };
                //Bank->addedNeutrons[idx].pos = Bank->addedNeutrons[idx].pos - bump;
                //Bank->addedNeutrons[idx].pos = randPos;
                //Bank->addedNeutrons[idx].dirVec = vec3::randomUnit(RNG);
                //seedNo[idx] = RNG.gen();
                //Bank->addedNeutrons[idx].Nullify();
                //atomicAdd(&(Bank->addedNeutronSize), -1);
                //return;
            }
        }
        //printf("idx %d neutron didn't reacted after %d loops..?\n", idx, iterTimes);
        seedNo[idx] = RNG.gen();
        return;
    }
    seedNo[idx] = RNG.gen();
    return;


}














H void cycle_Neutron_CPU(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, double& absorption, double& fission, double& leak) {
    for (int idx = 0; idx < Bank->allocatableNeutronNum; idx++) {
        GnuAMCM RNG(seedNo[idx]);
        double eps = globaleps;

        if (Bank->neutrons[idx].isNullified()) {
            seedNo[idx] = RNG.gen();
            continue;
        }

        for (int it = 0; it < 100; it++) {
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->neutrons[idx]);

            if (Bank->neutrons[idx].isNullified()) {
                Bank->neutronSize -= 1;
                seedNo[idx] = RNG.gen();
                break;
            }

            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->neutrons[idx]);
            vec3 flooredPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->neutrons[idx]);

            if (currentPincell.sideLength == 0.0 && currentPincell.height == 0.0) {
                Bank->neutrons[idx].Nullify();
                Bank->neutronSize -= 1;
                seedNo[idx] = RNG.gen();
                break;
            }

            double DTC = currentAssembly.DTC(Bank->neutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->neutrons[idx]);

            if (DTC < DTS) {
                Bank->neutrons[idx].updateWithLength(DTC);

                if (CoreTally) {
                    TallyAssembly& TA = CoreTally->returnAssemblyByNeutron(Bank->neutrons[idx]);
                    TallyPincell& TP = TA.returnPincellByPos(Bank->neutrons[idx]);
                    if (currentPincell.meatOrMod(flooredPos) == MatType::MOD) TP.modTally += DTC;
                    else TP.pinTally += DTC;
                }

                Interaction::reaction_CPU(idx, Bank->neutrons[idx], Bank, XSLib, currentPincell, flooredPos, RNG, k_mult, passFlag, false, absorption, fission);
                seedNo[idx] = RNG.gen();
                break;
            }
            else {
                if (CoreTally) {
                    TallyAssembly& TA = CoreTally->returnAssemblyByNeutron(Bank->neutrons[idx]);
                    TallyPincell& TP = TA.returnPincellByPos(Bank->neutrons[idx]);
                    if (currentPincell.meatOrMod(flooredPos) == MatType::MOD) TP.modTally += DTS;
                    else TP.pinTally += DTS;
                }

                vec3 updatedSurfacePos = Bank->neutrons[idx].pos + Bank->neutrons[idx].dirVec * DTS;

                if (updatedSurfacePos.x >= Core->x - eps || updatedSurfacePos.y >= Core->y - eps || updatedSurfacePos.z >= Core->z - eps) {
                    Bank->neutrons[idx].Nullify();
                    Bank->neutronSize -= 1;
                    leak += 1.0;
                    seedNo[idx] = RNG.gen();
                    break;
                }

                if (updatedSurfacePos.x <= eps || updatedSurfacePos.y <= eps || updatedSurfacePos.z <= eps) {
                    Interaction::reflection_CPU(Bank->neutrons[idx], DTS, updatedSurfacePos, eps);
                    if (Bank->neutrons[idx].isNullified()) {
                        Bank->neutronSize -= 1;
                        seedNo[idx] = RNG.gen();
                        break;
                    }
                    continue;
                }

                Bank->neutrons[idx].updateWithLength(DTS + eps);
            }
        }

        seedNo[idx] = RNG.gen();
    }
}

H void cycle_addedNeutron_CPU(NeutronBank* Bank, C5G7Geometry* Core, TallyC5G7Geometry* CoreTally, XSLibrary* XSLib, unsigned long long* seedNo, double* k_mult, bool passFlag, double& absorption, double& fission, double& leak) {
    for (int idx = 0; idx < Bank->allocatableNeutronNum; idx++) {
        GnuAMCM RNG(seedNo[idx]);
        double eps = globaleps;

        if (Bank->addedNeutrons[idx].isNullified()) {
            seedNo[idx] = RNG.gen();
            continue;
        }

        if (Bank->addedNeutrons[idx].passFlag) {
            Bank->addedNeutrons[idx].passFlag = false;
            seedNo[idx] = RNG.gen();
            continue;
        }

        for (int it = 0; it < 100; it++) {
            Assembly currentAssembly = Core->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);

            if (Bank->addedNeutrons[idx].isNullified()) {
                Bank->addedNeutronSize -= 1;
                seedNo[idx] = RNG.gen();
                break;
            }

            Pincell currentPincell = currentAssembly.returnPincellByPos(Bank->addedNeutrons[idx]);
            vec3 flooredPos = currentAssembly.returnFlooredNeutronPosInPincell(Bank->addedNeutrons[idx]);

            if (currentPincell.sideLength == 0.0 && currentPincell.height == 0.0) {
                Bank->addedNeutrons[idx].Nullify();
                Bank->addedNeutronSize -= 1;
                seedNo[idx] = RNG.gen();
                break;
            }

            double DTC = currentAssembly.DTC(Bank->addedNeutrons[idx], XSLib, RNG);
            double DTS = currentAssembly.DTS(Bank->addedNeutrons[idx]);

            if (DTC < DTS) {
                Bank->addedNeutrons[idx].updateWithLength(DTC);

                if (CoreTally) {
                    TallyAssembly& TA = CoreTally->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
                    TallyPincell& TP = TA.returnPincellByPos(Bank->addedNeutrons[idx]);
                    if (currentPincell.meatOrMod(flooredPos) == MatType::MOD) TP.modTally += DTC;
                    else TP.pinTally += DTC;
                }

                Interaction::reaction_CPU(idx, Bank->addedNeutrons[idx], Bank, XSLib, currentPincell, flooredPos, RNG, k_mult, passFlag, true, absorption, fission);
                seedNo[idx] = RNG.gen();
                break;
            }
            else {
                if (CoreTally) {
                    TallyAssembly& TA = CoreTally->returnAssemblyByNeutron(Bank->addedNeutrons[idx]);
                    TallyPincell& TP = TA.returnPincellByPos(Bank->addedNeutrons[idx]);
                    if (currentPincell.meatOrMod(flooredPos) == MatType::MOD) TP.modTally += DTS;
                    else TP.pinTally += DTS;
                }

                vec3 updatedSurfacePos = Bank->addedNeutrons[idx].pos + Bank->addedNeutrons[idx].dirVec * DTS;

                if (updatedSurfacePos.x >= Core->x - eps || updatedSurfacePos.y >= Core->y - eps || updatedSurfacePos.z >= Core->z - eps) {
                    Bank->addedNeutrons[idx].Nullify();
                    Bank->addedNeutronSize -= 1;
                    leak += 1.0;
                    seedNo[idx] = RNG.gen();
                    break;
                }

                if (updatedSurfacePos.x <= eps || updatedSurfacePos.y <= eps || updatedSurfacePos.z <= eps) {
                    Interaction::reflection_CPU(Bank->addedNeutrons[idx], DTS, updatedSurfacePos, eps);
                    if (Bank->addedNeutrons[idx].isNullified()) {
                        Bank->addedNeutronSize -= 1;
                        seedNo[idx] = RNG.gen();
                        break;
                    }
                    continue;
                }

                Bank->addedNeutrons[idx].updateWithLength(DTS + eps);
            }
        }

        seedNo[idx] = RNG.gen();
    }
}

H void addedNeutronPassResetter_CPU(NeutronBank* Bank) {
    for (int idx = 0; idx < Bank->allocatableNeutronNum; idx++) {
        if (!Bank->addedNeutrons[idx].isNullified()) {
            Bank->addedNeutrons[idx].passFlag = false;
        }
    }
}