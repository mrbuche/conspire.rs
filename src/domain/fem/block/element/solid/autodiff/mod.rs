#![allow(clippy::needless_range_loop)]

use crate::{
    math::{Current, TensorRank1List, TensorRank2List2D},
    units::{Force, ForcePerLength, ForcePerVelocity, Velocity},
};

pub(crate) type Forces<const D: usize, const N: usize> = TensorRank1List<D, Current, N, Force>;
pub(crate) type Stiffnesses<const D: usize, const N: usize> =
    TensorRank2List2D<D, Current, Current, N, N, ForcePerLength>;
pub(crate) type Velocities<const D: usize, const N: usize> =
    TensorRank1List<D, Current, N, Velocity>;
pub(crate) type Dampings<const D: usize, const N: usize> =
    TensorRank2List2D<D, Current, Current, N, N, ForcePerVelocity>;

pub(crate) fn component<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    x: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    let mut sum = 0.0;
    for a in 0..N {
        sum += x[D * a + i] * grad_n[base + D * a + j];
    }
    sum
}

fn entry<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    x: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    if i < D && j < D {
        component::<D, N, DOF, GN>(grad_n, base, x, i, j)
    } else if i == j {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn deformation_gradient<
    const D: usize,
    const N: usize,
    const DOF: usize,
    const GN: usize,
>(
    grad_n: &[f64; GN],
    g: usize,
    x: &[f64; DOF],
) -> [f64; 9] {
    let b = D * N * g;
    [
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 2),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 2),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 2),
    ]
}

fn rate_entry<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    v: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    if i < D && j < D {
        component::<D, N, DOF, GN>(grad_n, base, v, i, j)
    } else {
        0.0
    }
}

pub(crate) fn deformation_gradient_rate<
    const D: usize,
    const N: usize,
    const DOF: usize,
    const GN: usize,
>(
    grad_n: &[f64; GN],
    g: usize,
    v: &[f64; DOF],
) -> [f64; 9] {
    let b = D * N * g;
    [
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 2),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 2),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 2),
    ]
}

pub(crate) fn flatten_velocities<const D: usize, const N: usize, const DOF: usize>(
    velocities: &Velocities<D, N>,
) -> [f64; DOF] {
    let mut v = [0.0; DOF];
    for (a, velocity) in velocities.into_iter().enumerate() {
        for i in 0..D {
            v[D * a + i] = velocity[i].value();
        }
    }
    v
}

pub(crate) fn tangent_matrix(
    mut directional: impl FnMut(&[f64; 9], &mut [f64; 9], &mut [f64; 9]),
) -> [f64; 81] {
    let mut tangent = [0.0; 81];
    for kl in 0..9 {
        let mut direction = [0.0; 9];
        direction[kl] = 1.0;
        let (mut primal, mut seed) = ([0.0; 9], [0.0; 9]);
        directional(&direction, &mut primal, &mut seed);
        for ij in 0..9 {
            tangent[9 * ij + kl] = seed[ij];
        }
    }
    tangent
}

pub(crate) fn assemble_tangent<const D: usize, const N: usize, const G: usize, const GN: usize>(
    tangents: &[[f64; 81]; G],
    grad_n: &[f64; GN],
    weights: &[f64; G],
) -> [[[[f64; D]; D]; N]; N] {
    let mut out = [[[[0.0; D]; D]; N]; N];
    for g in 0..G {
        let base = D * N * g;
        for a in 0..N {
            for b in 0..N {
                for i in 0..D {
                    for k in 0..D {
                        let mut sum = 0.0;
                        for j in 0..D {
                            for l in 0..D {
                                sum += grad_n[base + D * a + j]
                                    * tangents[g][9 * (3 * i + j) + 3 * k + l]
                                    * grad_n[base + D * b + l];
                            }
                        }
                        out[a][b][i][k] += weights[g] * sum;
                    }
                }
            }
        }
    }
    out
}
