#include "cudaHeader.cuh"
#include "Neutron.cuh"

__host__ __device__ vec3 vec3::operator-(const vec3 vec) const {
	return { x - vec.x, y - vec.y, z - vec.z };
}
__host__ __device__ vec3 vec3::operator+(const vec3 vec) const {
	return { x + vec.x, y + vec.y, z + vec.z };
}
__host__ __device__ vec3 vec3::operator*(const double coeff) const {
	return { x * coeff, y * coeff, z * coeff };
}

__host__ __device__ vec3 vec3::operator/(const double coeff) const {
	return { x / coeff, y / coeff, z / coeff };
}
__host__ __device__ vec3 vec3::cross(const vec3 vec) const {
	return {
		y * vec.z - z * vec.y,
		z * vec.x - x * vec.z,
		x * vec.y - y * vec.x
	};
}
__host__ __device__ double vec3::dot(const vec3 vec) const {
	return x * vec.x + y * vec.y + z * vec.z;
}

__host__ __device__ double vec3::magnitude() const {
	return sqrt(x * x + y * y + z * z);
}

__host__ __device__ vec3 vec3::normalize() const {
	return {
		x / magnitude(),
		y / magnitude(),
		z / magnitude()
	};
}


__host__ __device__ vec3 vec3::randomUnit(GnuAMCM& RNG) {	// static
	// wrong!!!!
	//vec3 randUnitVec = { static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()) };
	//vec3 randUnitVec = { localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0) };

	double phi = RNG.uniform(0, 1) * 2 * Constants::PI;
	double theta = acos(2 * RNG.uniform(0, 1) - 1);
	Spherical sphericalDir(theta, phi, AngleType::Radian);
	return sphericalDir.convToVec3();
}


__host__ __device__  Spherical vec3::convToSpherical() const {
	double theta, phi;
	if (this->magnitude() == 1) {
		theta = acos(z);
		phi = atan2(y, x);
	}
	else {
		vec3 normalizedVec = this->normalize();
		theta = acos(normalizedVec.z);
		phi = atan2(normalizedVec.y, normalizedVec.x);
	}

	return { theta, phi, AngleType::Radian, 0 };
}

__host__ __device__ double Neutron::Velocity() const {
	return sqrt(2 * energy * Constants::ElectronC / (Constants::M_Neutron * Constants::amuToKilogram));
	// Constants namespace's atom mass always have gram/mol (i.e. amu) - convert it to kg
}

__host__ __device__ vec3 Neutron::VelocityVec() const {
	return dirVec * this->Velocity();
}

__host__ __device__ void Neutron::Nullify() {
	this->pos = { 0.0, 0.0, 0.0 };
	this->dirVec = { 0.0, 0.0, 0.0 };
	this->energy = 0.0;
	this->status = false;
	this->passFlag = true;
	this->weight = 0.0;
}

__host__ __device__ void Neutron::reInitialize(vec3 pos, vec3 dirVec, double energy, double weight, bool passFlag) {
	this->pos = pos;
	this->dirVec = dirVec;
	this->energy = energy;
	this->status = true;
	this->weight = weight;
	this->passFlag = passFlag;
}

__host__ __device__ bool Neutron::isNullified() const {
	if (this->status) { return false; }
	else { return true; }
}

__host__ __device__ void Neutron::updateWithLength(double length) {
	this->pos.x += length * this->dirVec.x;
	this->pos.y += length * this->dirVec.y;
	this->pos.z += length * this->dirVec.z;
}


HD void Neutron::setWeight(double weight) {
	this->weight = weight;
}


__host__ __device__ vec3 Spherical::convToVec3() const {
	double x = r * sin(this->theta) * cos(this->phi);
	double y = r * sin(this->theta) * sin(this->phi);
	double z = r * cos(this->theta);

	return { x, y, z };
}
