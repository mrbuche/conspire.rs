//! Automatic differentiation of hyperelastic energies via `std::autodiff` (Enzyme).
//!
//! Opt-in, `--features autodiff`. Needs a nightly `rustc` with the Enzyme
//! backend (`rustup component add enzyme`) and a fat-LTO profile:
//!
//! ```text
//! RUSTFLAGS="-Zautodiff=Enable" cargo +nightly test --release --features autodiff -j1
//! ```
//!
//! Maintained models keep their hand-written stress and tangent. This module is
//! for prototyping a new model from its energy alone, and for cross-checking the
//! hand-written derivatives in tests.
//!
//! Deformation gradients are flattened row-major (`f[3 * i + j] = F_iJ`).
//! Reverse mode over the energy gives `P_iJ = dPsi/dF_iJ`; forward mode over a
//! stress kernel gives that stress's `d(stress)/dF` tangent.

#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

pub mod neo_hookean;

use crate::mechanics::DeformationGradient;

fn flatten(deformation_gradient: &DeformationGradient) -> [f64; 9] {
    let mut f = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            f[3 * i + j] = deformation_gradient[i][j].value();
        }
    }
    f
}

fn unflatten_stress<T: From<[[f64; 3]; 3]>>(m: [f64; 9]) -> T {
    T::from([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]])
}

fn unflatten_tangent<T: From<[[[[f64; 3]; 3]; 3]; 3]>>(c: [[[[f64; 3]; 3]; 3]; 3]) -> T {
    T::from(c)
}
