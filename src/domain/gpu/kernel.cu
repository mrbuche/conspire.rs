// The nodal forces of the elements of a block of a Neo-Hookean solid.
//
// REAL is the precision, NODES the nodes of an element, and POINTS its integration points, all
// of which are defined when the kernels are compiled. The kernels take everything that is
// particular to an element as data, so they serve every element with the three dimensions: a
// block hands them the gradients of the shape functions at its integration points, which the
// elements keep, and the weights that go with them.
//
// The kernels do what the elements of the library do, in the same order, so that the results
// are those of the library to the precision: the forces of an element are written first, and
// each node then adds the forces that it is given, in the order of the elements.
//
// Everything that is per element is laid out with the element last, so that the threads of a
// warp, which are consecutive elements, read and write consecutive addresses:
//   connectivity[node * items + item]
//   reference[value * items + item], where the values of an element are the gradient vectors of
//     its nodes at each of its integration points (point, then node, then component), and then
//     the weights of the points, REFERENCE_SIZE of them
//   forces[(component * NODES + node) * items + item]

#define REFERENCE_SIZE (POINTS * NODES * 3 + POINTS)

// A value of the reference data of the element of the thread.
#define REFERENCE(value) reference[(size_t)(value) * items + item]

extern "C" __global__ void item_forces(
    const unsigned int items,
    const unsigned int *__restrict__ connectivity,
    const REAL *__restrict__ reference,
    const REAL *__restrict__ coordinates,
    REAL *__restrict__ forces,
    unsigned int *__restrict__ failure,
    const REAL shear_modulus,
    const REAL bulk_modulus)
{
    const unsigned int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items)
    {
        return;
    }
    REAL x[NODES][3];
    REAL f[NODES][3];
#pragma unroll
    for (int a = 0; a < NODES; ++a)
    {
        const size_t node = connectivity[(size_t)a * items + item];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            x[a][i] = coordinates[3 * node + i];
            f[a][i] = 0;
        }
    }
    bool valid = true;
    for (int g = 0; g < POINTS && valid; ++g)
    {
        // The deformation gradient, the sum of the outer products of the coordinates and the
        // gradient vectors of the nodes.
        REAL F[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                F[i][j] = 0;
            }
        }
#pragma unroll
        for (int a = 0; a < NODES; ++a)
        {
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
#pragma unroll
                for (int j = 0; j < 3; ++j)
                {
                    F[i][j] += x[a][i] * REFERENCE((g * NODES + a) * 3 + j);
                }
            }
        }
        // The cofactors, which are the Jacobian times the inverse transpose.
        REAL cofactor[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            const int i1 = (i + 1) % 3;
            const int i2 = (i + 2) % 3;
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                const int j1 = (j + 1) % 3;
                const int j2 = (j + 2) % 3;
                cofactor[i][j] = F[i1][j1] * F[i2][j2] - F[i1][j2] * F[i2][j1];
            }
        }
        const REAL jacobian = F[0][0] * cofactor[0][0] + F[0][1] * cofactor[0][1] + F[0][2] * cofactor[0][2];
        if (!(jacobian > 0))
        {
            // The first item that fails is the one that is reported, whichever thread gets there.
            atomicMin(failure, item);
            valid = false;
            break;
        }
        // The Cauchy stress.
        REAL b[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                b[i][j] = F[i][0] * F[j][0] + F[i][1] * F[j][1] + F[i][2] * F[j][2];
            }
        }
        const REAL third_of_trace = (b[0][0] + b[1][1] + b[2][2]) / 3;
        const REAL scaling = shear_modulus / pow(jacobian, (REAL)(5.0 / 3.0));
        const REAL pressure = bulk_modulus * (REAL)0.5 * (jacobian - 1 / jacobian);
        REAL sigma[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                const REAL deviatoric = b[i][j] - (i == j ? third_of_trace : 0);
                sigma[i][j] = deviatoric * scaling + (i == j ? pressure : 0);
            }
        }
        // The first Piola-Kirchhoff stress, the Cauchy stress times the cofactors.
        REAL P[3][3];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
#pragma unroll
            for (int j = 0; j < 3; ++j)
            {
                P[i][j] = sigma[i][0] * cofactor[0][j] + sigma[i][1] * cofactor[1][j] + sigma[i][2] * cofactor[2][j];
            }
        }
        const REAL weight = REFERENCE(POINTS * NODES * 3 + g);
#pragma unroll
        for (int a = 0; a < NODES; ++a)
        {
            const int gradient = (g * NODES + a) * 3;
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                f[a][i] += (P[i][0] * REFERENCE(gradient) + P[i][1] * REFERENCE(gradient + 1) + P[i][2] * REFERENCE(gradient + 2)) * weight;
            }
        }
    }
#pragma unroll
    for (int a = 0; a < NODES; ++a)
    {
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            forces[((size_t)i * NODES + a) * items + item] = valid ? f[a][i] : 0;
        }
    }
}

// The forces on the nodes, which each add those of the elements that have them, in the order of
// the elements. A contribution is `position of the node in the element * items + item`, which is
// where the force of the element on the node is in each plane of components.
extern "C" __global__ void gather(
    const unsigned int nodes,
    const unsigned int items,
    const unsigned int *__restrict__ offsets,
    const unsigned int *__restrict__ contributions,
    const REAL *__restrict__ forces,
    REAL *__restrict__ nodal_forces)
{
    const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes)
    {
        return;
    }
    const size_t plane = (size_t)NODES * items;
    REAL sum[3] = {nodal_forces[3 * (size_t)node], nodal_forces[3 * (size_t)node + 1], nodal_forces[3 * (size_t)node + 2]};
    for (unsigned int k = offsets[node]; k < offsets[node + 1]; ++k)
    {
        const size_t contribution = contributions[k];
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            sum[i] += forces[i * plane + contribution];
        }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        nodal_forces[3 * (size_t)node + i] = sum[i];
    }
}
