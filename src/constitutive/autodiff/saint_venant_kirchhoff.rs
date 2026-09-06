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
    let mut e = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut c_ij = 0.0;
            for k in 0..3 {
                c_ij += f[3 * k + i] * f[3 * k + j];
            }
            e[3 * i + j] = 0.5 * (c_ij - if i == j { 1.0 } else { 0.0 });
        }
    }
    let trace_e = e[0] + e[4] + e[8];
    let mut squared_trace_e = 0.0;
    for k in 0..9 {
        squared_trace_e += e[k] * e[k];
    }
    shear_modulus * squared_trace_e
        + 0.5 * (bulk_modulus - 2.0 / 3.0 * shear_modulus) * trace_e * trace_e
}

/// `P_iJ = dPsi/dF_iJ`, row-major, by reverse-mode AD of [`energy`].
#[autodiff_forward(d_piola, Const, Const, Dual, Dual)]
fn piola(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9], out: &mut [f64; 9]) {
    *out = [0.0; 9];
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
