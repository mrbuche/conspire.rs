use super::{flatten, unflatten_stress, unflatten_tangent};
use crate::{
    constitutive::solid::{Solid, hyperelastic::NeoHookean},
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffTangentStiffness,
    },
};
use std::autodiff::{autodiff_forward, autodiff_reverse};

/// Helmholtz free energy density, `f` row-major.
///
/// Mirrors `<NeoHookean as Hyperelastic>::helmholtz_free_energy_density`.
#[autodiff_reverse(d_energy, Const, Const, Duplicated, Active)]
fn energy(bulk_modulus: f64, shear_modulus: f64, f: &[f64; 9]) -> f64 {
    let mut trace_b = 0.0;
    for k in 0..9 {
        trace_b += f[k] * f[k];
    }
    let jacobian = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6])
        + f[2] * (f[3] * f[7] - f[4] * f[6]);
    0.5 * (shear_modulus * (trace_b * jacobian.powf(-2.0 / 3.0) - 3.0)
        + bulk_modulus * (0.5 * (jacobian * jacobian - 1.0) - jacobian.ln()))
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
    model: &NeoHookean,
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
    model: &NeoHookean,
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
