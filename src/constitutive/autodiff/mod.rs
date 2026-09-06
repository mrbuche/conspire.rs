//! Automatic differentiation of hyperelastic energies via `std::autodiff` (Enzyme).
//!
//! Opt-in, `--features autodiff`. Needs a nightly `rustc` built with the Enzyme
//! backend and a fat-LTO profile:
//!
//! ```text
//! RUSTFLAGS="-Zautodiff=Enable" cargo +nightly test --release --features autodiff
//! ```
//!
//! Maintained models keep their hand-written stress and tangent. This module is
//! for prototyping a new model from its energy alone, and for cross-checking the
//! hand-written derivatives in tests.
//!
//! Deformation gradients are flattened row-major (`f[3 * i + j] = F_iJ`).
//! Reverse mode over the energy gives `P_iJ = dPsi/dF_iJ`; forward-over-reverse
//! gives `C_iJkL = dP_iJ/dF_kL`.

#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

pub mod neo_hookean;

use crate::mechanics::{
    DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
};

fn flatten(deformation_gradient: &DeformationGradient) -> [f64; 9] {
    let mut f = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            f[3 * i + j] = deformation_gradient[i][j].value();
        }
    }
    f
}

fn unflatten_stress(p: [f64; 9]) -> FirstPiolaKirchhoffStress {
    FirstPiolaKirchhoffStress::from([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]])
}

fn unflatten_tangent(c: [[[[f64; 3]; 3]; 3]; 3]) -> FirstPiolaKirchhoffTangentStiffness {
    FirstPiolaKirchhoffTangentStiffness::from(c)
}
