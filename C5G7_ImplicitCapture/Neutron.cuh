#pragma once
#include "cudaHeader.cuh"

enum AngleType {
	Degree,
	Radian
};

struct Spherical;

struct vec2 {
	double x, y; // CM!!!

	HD vec2() : x(0), y(0) {}
	HD vec2(double x, double y) : x(x), y(y) {}

};

struct vec3 {
	double x, y, z;

	__host__ __device__ vec3()
		: x(0.0), y(0.0), z(0.0)
	{
	}

	__host__ __device__ vec3(double xx, double yy, double zz)
		: x(xx), y(yy), z(zz)
	{
	}


	__host__ __device__ vec3 operator-(const vec3 vec) const;
	__host__ __device__ vec3 operator+(const vec3 vec) const;
	__host__ __device__ vec3 operator*(const double coeff) const;
	__host__ __device__ vec3 operator/(const double coeff) const;

	__host__ __device__ vec3 cross(const vec3 vec) const;
	__host__ __device__ double dot(const vec3 vec) const;

	__host__ __device__ double magnitude() const;

	__host__ __device__ vec3 normalize() const;

	// this is static - there's no need to make a void function for randomizing vectors.
	__host__ __device__ static vec3 randomUnit(GnuAMCM& RNG);

	/*__host__ __device__ static vec3 randomUnit(GnuAMCM& RNG) {	// static

		//vec3 randUnitVec = { static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()) };
		//vec3 randUnitVec = { localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0) };

		double phi = RNG.uniform(0, 1) * 2 * Constants::PI;
		double theta = acos(2 * RNG.uniform(0, 1) - 1);
		Spherical sphericalDir(theta, phi, AngleType::Radian);
		return sphericalDir.convToVec3();
	}
	*/

	__host__ __device__ Spherical convToSpherical() const;
};

struct Spherical {
	double theta, phi;	// always in radian
	double r;
	AngleType angleType;


	__host__ __device__ Spherical()
		: theta(0.0), phi(0.0), r(0.0), angleType(AngleType::Radian)
	{
	}

	__host__ __device__ Spherical(double theta, double phi, AngleType angleType, double r = 1.0)
		: theta(theta), phi(phi), angleType(angleType), r(r)
	{
		if (angleType == AngleType::Degree) {
			// Degree to radian
			this->theta = this->theta * atan(1.0) * 4.0 / 180.0;
			this->phi = this->phi * atan(1.0) * 4.0 / 180.0;
			//this->theta = this->theta * Constants::PI / 180.0;
			//this->phi = this->phi * Constants::PI / 180.0;
			this->angleType = AngleType::Radian;
		}
	}

	__host__ __device__ vec3 convToVec3() const;
};

struct Neutron {
	vec3 pos;			// CENTI METERS !!!
	vec3 dirVec;		// UNIT VECTOR
	double energy;		// For this, we will use as Groups - 1.0 = group 1, so on.
	bool status;
	bool passFlag;
	double weight;

	__host__ __device__ Neutron()
		: pos({ 0.0, 0.0, 0.0 }), dirVec({ 0.0, 0.0, 0.0 }), energy(0.0), status(false), passFlag(true), weight(0.0) 
	{
	}

	__host__ __device__ Neutron(vec3 pos, vec3 dirVec, double energy, double weight = 1.0)
		: pos(pos), dirVec(dirVec), energy(energy), status(true), passFlag(false), weight(weight)
	{
	}

	__host__ __device__ Neutron(vec3 pos, Spherical sphercial, double energy, double weight = 1.0)
		: pos(pos), dirVec(sphercial.convToVec3()), energy(energy), status(true), passFlag(false), weight(weight)
	{
	}

	__host__ __device__ double Velocity() const;
	__host__ __device__ vec3 VelocityVec() const;

	__host__ __device__ void Nullify();
	__host__ __device__ bool isNullified() const;
	__host__ __device__ void reInitialize(vec3 pos, vec3 dir, double energy, double weight, bool passFlag);
	

	__host__ __device__ void updateWithLength(double length);
	//HD vec3 pos

	HD void setWeight(double weight);

	__host__ inline void printInfo() {
		std::cout << "(" << this->pos.x << ", " << this->pos.y << ", " << this->pos.z << "),  ";
		std::cout << "(" << this->dirVec.x << ", " << this->dirVec.y << ", " << this->dirVec.z << ") , status: ";
		if (this->status) { std::cout << " true.  "; }
		else { std::cout << " false.  "; }

		std::cout << "passFlag: ";
		if (this->passFlag) { std::cout << "true.\n"; }
		else { std::cout << "false.\n"; }
	}


	__device__ inline void printInfo_Kernel(int idx) {
		const char* str_status;
		const char* str_passFlag;
		if (this->status) { str_status = "true"; }
		else { str_status = "false"; }

		if (this->passFlag) { str_passFlag = "true"; }
		else { str_passFlag = "false"; }

		printf("index %d: ( %.5f, %.5f, %.5f ), ( %.5f, %.5f, %.5f ), status: %s, passflag: %s\n",
			idx, this->pos.x, this->pos.y, this->pos.z, this->dirVec.x, this->dirVec.y, this->dirVec.z, str_status, str_passFlag);
	}

};

struct NeutronBank {
	Neutron* neutrons;
	Neutron* addedNeutrons;
	int neutronSize;
	int allocatableNeutronNum;
	int addedNeutronSize;
	int addedNeutronIndex;
	unsigned long long seedNo;

	__host__ __device__ NeutronBank(unsigned int bankSize, unsigned int neutronNum, unsigned long long seedNo)
		: neutrons(new Neutron[bankSize]), addedNeutrons(new Neutron[bankSize]), neutronSize(neutronNum), allocatableNeutronNum(bankSize),
		addedNeutronSize(0), addedNeutronIndex(0), seedNo(seedNo)
	{
	}


	__host__ __device__ ~NeutronBank() {
		// Note this delete[] operation will also deallocate the device side shits:
		// you must nullptr a temporary objects containing device pointers.
		// this will be manually done at the end of the main file
		//delete[] neutrons;
		//delete[] addedNeutrons;
	}

	__host__ __device__ inline int getTotalNeutronNum() {
		return this->addedNeutronSize + this->neutronSize;
	}



};

