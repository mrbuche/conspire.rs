use super::{flatten, unflatten_stress, unflatten_tangent};
use crate::{
    constitutive::solid::{Solid, hyperelastic::SaintVenantKirchhoff},
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
    },
};
use std::autodiff::{autodiff_forward, autodiff_reverse};

/// Helmholtz free energy density, `f` row-major.
///
/// Mirrors `<SaintVenantKirchhoff as Hyperelastic>::helmholtz_free_energy_density`:
/// `E = (F^T F - I) / 2`, `Psi = mu tr(E^2) + (kappa - 2 mu / 3) (tr E)^2 / 2`.
#[autodiff_reverse(d_energy, Const, Const, Duplicated, Active)]
fn energy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9]) -> f64 {
    // C = F^T F, kept as scalars (no memset'd buffer for Enzyme to mis-type).
    let c00 = f[0] * f[0] + f[3] * f[3] + f[6] * f[6];
    let c11 = f[1] * f[1] + f[4] * f[4] + f[7] * f[7];
    let c22 = f[2] * f[2] + f[5] * f[5] + f[8] * f[8];
    let c01 = f[0] * f[1] + f[3] * f[4] + f[6] * f[7];
    let c02 = f[0] * f[2] + f[3] * f[5] + f[6] * f[8];
    let c12 = f[1] * f[2] + f[4] * f[5] + f[7] * f[8];
    let trace_c = c00 + c11 + c22;
    let trace_e = 0.5 * (trace_c - 3.0);
    // tr(E^2) = (sum C_ij^2 - 2 tr C + 3) / 4
    let squared_trace_e = 0.25
        * (c00 * c00 + c11 * c11 + c22 * c22 + 2.0 * (c01 * c01 + c02 * c02 + c12 * c12)
            - 2.0 * trace_c
            + 3.0);
    shear_modulus * squared_trace_e
        + 0.5 * (bulk_modulus - 2.0 / 3.0 * shear_modulus) * trace_e * trace_e
}

/// `P_iJ = dPsi/dF_iJ`, row-major, by reverse-mode AD of [`energy`].
#[autodiff_forward(d_piola, Const, Const, Dual, Dual)]
fn piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    for out_i in out.iter_mut() {
        *out_i = 0.0;
    }
    d_energy(bulk_modulus, shear_modulus, f, out, 1.0);
}

/// First Piola-Kirchhoff stress, reverse-mode AD of the energy.
pub fn first_piola_kirchhoff_stress(
    model: &SaintVenantKirchhoff,
    deformation_gradient: &DeformationGradient,
) -> FirstPiolaKirchhoffStress {
    let f = flatten(deformation_gradient);
    let mut p = [0.0; 9];
    piola(
        model.bulk_modulus().value(),
        model.shear_modulus().value(),
        &f,
        &mut p,
    );
    unflatten_stress(p)
}

/// First Piola-Kirchhoff tangent stiffness, forward-over-reverse AD of the energy.
pub fn first_piola_kirchhoff_tangent_stiffness(
    model: &SaintVenantKirchhoff,
    deformation_gradient: &DeformationGradient,
) -> FirstPiolaKirchhoffTangentStiffness {
    let (bulk_modulus, shear_modulus) =
        (model.bulk_modulus().value(), model.shear_modulus().value());
    let f = flatten(deformation_gradient);
    let mut c = [[[[0.0; 3]; 3]; 3]; 3];
    for k in 0..3 {
        for l in 0..3 {
            let mut df = [0.0; 9];
            df[3 * k + l] = 1.0;
            let (mut p, mut dp) = ([0.0; 9], [0.0; 9]);
            d_piola(bulk_modulus, shear_modulus, &f, &df, &mut p, &mut dp);
            for i in 0..3 {
                for j in 0..3 {
                    c[i][j][k][l] = dp[3 * i + j];
                }
            }
        }
    }
    unflatten_tangent(c)
}
