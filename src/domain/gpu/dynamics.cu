// The steps of velocity Verlet on the degrees of freedom of a solid, with the coordinates,
// velocities and accelerations kept on the device.
//
// REAL is the precision, which is defined when the kernels are compiled. The kernels do what the
// integrator of the library does, in the same order, and the masses of held degrees of freedom
// are infinite so that they do not accelerate.

// The first half of a step: the velocity a half step on, and the coordinates a step on with it.
extern "C" __global__ void kick_drift(
    const unsigned int count,
    REAL *__restrict__ coordinates,
    REAL *__restrict__ velocities,
    const REAL *__restrict__ accelerations,
    const REAL dt)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const REAL half_velocity = accelerations[i] * (dt * (REAL)0.5) + velocities[i];
    velocities[i] = half_velocity;
    coordinates[i] = half_velocity * dt + coordinates[i];
}

// The accelerations that balance the forces, which are then cleared for the next step, and
// the velocity another half step on when the step is not the first evaluation.
extern "C" __global__ void accelerate(
    const unsigned int count,
    REAL *__restrict__ forces,
    const REAL *__restrict__ masses,
    REAL *__restrict__ accelerations,
    REAL *__restrict__ velocities,
    const REAL dt,
    const unsigned int kick)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const REAL acceleration = -forces[i] / masses[i];
    forces[i] = 0;
    accelerations[i] = acceleration;
    if (kick)
    {
        velocities[i] = acceleration * (dt * (REAL)0.5) + velocities[i];
    }
}
