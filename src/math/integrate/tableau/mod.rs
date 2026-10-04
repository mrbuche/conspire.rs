#[cfg(test)]
mod test;

use crate::math::Scalar;

/// Butcher tableau for an explicit Runge–Kutta method.
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
    /// Whether the last stage of a step is the first stage of the next.
    const FSAL: bool = false;
    /// Extents of the stability region along the negative real and the imaginary axes.
    fn stability() -> StabilityInterval {
        let mut coefficients = vec![1.0];
        let mut v = vec![1.0; Self::STAGES];
        for _ in 0..Self::STAGES {
            coefficients.push(Self::B.iter().zip(v.iter()).map(|(b, v)| b * v).sum());
            v = (0..Self::STAGES)
                .map(|i| Self::A[i].iter().zip(v.iter()).map(|(a, v)| a * v).sum())
                .collect();
        }
        StabilityInterval {
            real: extent(|s| amplification(&coefficients, -s, 0.0)),
            imaginary: extent(|s| amplification(&coefficients, 0.0, s)),
        }
    }
}

/// How far along each axis the stability region of a method extends.
#[derive(Clone, Copy, Debug)]
pub struct StabilityInterval {
    /// Extent along the negative real axis.
    pub real: Scalar,
    /// Extent along the imaginary axis.
    pub imaginary: Scalar,
}

const STABILITY_STEP: Scalar = 1e-3;
const STABILITY_MAX: Scalar = 200.0;
const STABILITY_TOL: Scalar = 1e-14;

fn amplification(coefficients: &[Scalar], re: Scalar, im: Scalar) -> Scalar {
    let (mut r, mut i) = (0.0, 0.0);
    for c in coefficients.iter().rev() {
        (r, i) = (r * re - i * im + c, r * im + i * re);
    }
    r * r + i * i
}

fn extent(squared_amplification: impl Fn(Scalar) -> Scalar) -> Scalar {
    let stable = |s| squared_amplification(s) <= 1.0 + STABILITY_TOL;
    let mut hi = STABILITY_STEP;
    while stable(hi) {
        if hi >= STABILITY_MAX {
            return STABILITY_MAX;
        }
        hi += STABILITY_STEP;
    }
    if hi == STABILITY_STEP {
        return 0.0;
    }
    let mut lo = hi - STABILITY_STEP;
    while hi - lo > STABILITY_TOL {
        let mid = 0.5 * (lo + hi);
        if stable(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// An embedded explicit Runge–Kutta pair.
pub trait EmbeddedTableau: ButcherTableau {
    /// Difference of the propagating and embedded weights.
    const D: &'static [Scalar];
}
