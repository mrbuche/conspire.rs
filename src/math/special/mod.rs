#[cfg(test)]
mod test;

use crate::{ABS_TOL, REL_TOL};
use std::f64::consts::E;

const NINE_FIFTHS: f64 = 9.0 / 5.0;

/// Returns the inverse Langevin function.
///
/// ```math
/// x = \mathcal{L}^{-1}(y)
/// ```
/// The first few terms of the Maclaurin series are used when $`|y|<3e^{-3}`$.
/// ```math
/// x \sim 3y + \frac{9}{5}y^3 + \mathrm{ord}(y^5)
/// ```
/// Two iterations of Newton's method are used to improve upon an initial guess given by [inverse_langevin_approximate()] otherwise.
/// The resulting maximum relative error is below $`1e^{-12}`$.
pub fn inverse_langevin(y: f64) -> f64 {
    let y_abs = y.abs();
    if y_abs >= 1.0 {
        panic!()
    } else if y_abs <= 3e-3 {
        3.0 * y + NINE_FIFTHS * y.powi(3)
    } else {
        let mut x = inverse_langevin_approximate(y_abs);
        for _ in 0..2 {
            x += (y_abs - langevin(x)) / langevin_derivative(x);
        }
        if y < 0.0 {
            -x
        } else {
            x
        }
    }
}

/// Returns an approximation of the inverse Langevin function.[^cite]
///
/// [^cite]: R. Jedynak, [Math. Mech. Solids **24**, 1992 (2019)](https://doi.org/10.1177/1081286518811395).
///
/// ```math
/// \mathcal{L}^{-1}(y) \approx \frac{2.14234 y^3 - 4.22785 y^2 + 3y}{(1 - y)(0.71716 y^3 - 0.41103 y^2 - 0.39165 y + 1)}
/// ```
/// This approximation has a maximum relative error of $`8.2e^{-4}`$.
pub fn inverse_langevin_approximate(y: f64) -> f64 {
    (2.14234 * y.powi(3) - 4.22785 * y.powi(2) + 3.0 * y)
        / (1.0 - y)
        / (0.71716 * y.powi(3) - 0.41103 * y.powi(2) - 0.39165 * y + 1.0)
}

/// Returns the Lambert W function.
///
/// ```math
/// y = W_0(x)
/// ```
pub fn lambert_w(x: f64) -> f64 {
    if x == -1.0 / E {
        -1.0
    } else if x == 0.0 {
        0.0
    } else if x == E {
        1.0
    } else if x < -1.0 / E {
        panic!()
    } else {
        let mut w = if x < 0.0 {
            (E * x * (1.0 + (1.0 + E * x).sqrt()).ln()) / (1.0 + E * x + (1.0 + E * x).sqrt())
        } else if x < E {
            x / E
        } else {
            x.ln() - x.ln().ln()
        };
        let mut error = w * w.exp() - x;
        while error.abs() >= ABS_TOL && (error / x).abs() >= REL_TOL {
            w *= (1.0 + (x / w).ln()) / (1.0 + w);
            error = w * w.exp() - x;
        }
        w
    }
}

/// Returns the Langevin function.
///
/// ```math
/// \mathcal{L}(x) = \coth(x) - x^{-1}
/// ```
pub fn langevin(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        1.0 / x.tanh() - 1.0 / x
    }
}

/// Returns derivative of the Langevin function.
///
/// ```math
/// \mathcal{L}'(x) = x^{-2} - \sinh^{-2}(x)
/// ```
pub fn langevin_derivative(x: f64) -> f64 {
    1.0 / x.powi(2) - 1.0 / x.sinh().powi(2)
}
