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
