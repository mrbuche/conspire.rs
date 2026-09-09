#[cfg(test)]
mod test;

use crate::math::Scalar;

/// Butcher tableau for an explicit Runge–Kutta method.
///
/// `A` is strictly lower triangular, stored ragged: row `i` holds `a[i][0..i]`.
pub trait ButcherTableau {
    /// Number of stages, equal to the number of stored slopes.
    const STAGES: usize;
    /// Order of the propagating solution; the exponent for adaptive time steps.
    const ORDER: Scalar;
    /// Stage-coupling coefficients, `A[i].len() == i`.
    const A: &'static [&'static [Scalar]];
    /// Stage abscissae, `C[0] == 0`.
    const C: &'static [Scalar];
    /// Propagating weights.
    const B: &'static [Scalar];
}

/// An embedded explicit Runge–Kutta pair.
///
/// The local error estimate is `dt * Σ D[i] k[i]`, with `D[i] = B[i] - B̂[i]`.
pub trait EmbeddedTableau: ButcherTableau {
    /// Difference of the propagating and embedded weights.
    const D: &'static [Scalar];
    /// Whether the last stage of an accepted step is the first stage of the next.
    const FSAL: bool = false;
}

/// Bogacki–Shampine 3(2) pair.
#[derive(Debug)]
pub struct BogackiShampine32;

impl ButcherTableau for BogackiShampine32 {
    const STAGES: usize = 4;
    const ORDER: Scalar = 3.0;
    const A: &'static [&'static [Scalar]] = &[
        &[],
        &[0.5],
        &[0.0, 0.75],
        &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
    ];
    const C: &'static [Scalar] = &[0.0, 0.5, 0.75, 1.0];
    const B: &'static [Scalar] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];
}

impl EmbeddedTableau for BogackiShampine32 {
    const D: &'static [Scalar] = &[-5.0 / 72.0, 6.0 / 72.0, 8.0 / 72.0, -9.0 / 72.0];
    const FSAL: bool = true;
}

/// Dormand–Prince 5(4) pair.
#[derive(Debug)]
pub struct DormandPrince54;

impl ButcherTableau for DormandPrince54 {
    const STAGES: usize = 7;
    const ORDER: Scalar = 5.0;
    const A: &'static [&'static [Scalar]] = &[
        &[],
        &[0.2],
        &[0.075, 0.225],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        &[
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        &[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];
    const C: &'static [Scalar] = &[0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];
    const B: &'static [Scalar] = &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];
}

impl EmbeddedTableau for DormandPrince54 {
    const D: &'static [Scalar] = &[
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -0.025,
    ];
    const FSAL: bool = true;
}
