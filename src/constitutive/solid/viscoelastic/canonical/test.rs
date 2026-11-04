use super::*;
use crate::{
    constitutive::{fluid::hyperviscous::Newtonian, solid::elastic::AlmansiHamelEulerian},
    math::{
        TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
    },
    units::{Rate, Stress, Viscosity},
};

#[test]
fn the_stress_measures_agree() -> Result<(), AssertionError> {
    let deformation_gradient = deformation_gradient();
    let deformation_gradient_rate = deformation_gradient_rate();
    let model = model();
    let cauchy = model.cauchy_stress(&deformation_gradient, &deformation_gradient_rate)?;
    let first =
        model.first_piola_kirchhoff_stress(&deformation_gradient, &deformation_gradient_rate)?;
    let second =
        model.second_piola_kirchhoff_stress(&deformation_gradient, &deformation_gradient_rate)?;
    Assert::default().eq_within_tols(
        &first,
        &(cauchy * deformation_gradient.inverse_transpose() * deformation_gradient.determinant()),
    )?;
    Assert::default().eq_within_tols(&second, &(deformation_gradient.inverse() * first))
}

fn model() -> Canonical<AlmansiHamelEulerian, Newtonian> {
    Canonical::from((
        AlmansiHamelEulerian {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        Newtonian {
            bulk_viscosity: Viscosity::pascal_seconds(11.0),
            shear_viscosity: Viscosity::pascal_seconds(1.0),
        },
    ))
}

fn deformation_gradient() -> DeformationGradient {
    DeformationGradient::from([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ])
}

fn deformation_gradient_rate() -> DeformationGradientRate {
    DeformationGradientRate::from([
        [0.16276008, 0.16544806, 0.10516932],
        [0.11349288, 0.16559786, 0.13899089],
        [0.19497108, 0.11119965, 0.19226318],
    ])
}

#[test]
fn finite_difference() -> Result<(), AssertionError> {
    let deformation_gradient = deformation_gradient();
    let deformation_gradient_rate = deformation_gradient_rate();
    let model = model();
    let tangent =
        model.cauchy_rate_tangent_stiffness(&deformation_gradient, &deformation_gradient_rate)?;
    let mut fd = CauchyRateTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut rate_plus = deformation_gradient_rate.clone();
            rate_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let stress_plus = model.cauchy_stress(&deformation_gradient, &rate_plus)?;
            let mut rate_minus = deformation_gradient_rate.clone();
            rate_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let stress_minus = model.cauchy_stress(&deformation_gradient, &rate_minus)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (stress_plus[i][j] - stress_minus[i][j])
                        / perturbation::<Rate>(crate::EPSILON);
                }
            }
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent, &fd)
    } else {
        Ok(())
    }
}

mod tangent {
    use super::*;
    use crate::{
        constitutive::fluid::viscous::test::finite_difference_tangent,
        mechanics::{
            CauchyTangentStiffness, FirstPiolaKirchhoffTangentStiffness,
            SecondPiolaKirchhoffTangentStiffness,
        },
    };
    finite_difference_tangent!(
        cauchy,
        cauchy_stress,
        cauchy_tangent_stiffness,
        CauchyTangentStiffness
    );
    finite_difference_tangent!(
        first_piola_kirchhoff,
        first_piola_kirchhoff_stress,
        first_piola_kirchhoff_tangent_stiffness,
        FirstPiolaKirchhoffTangentStiffness
    );
    finite_difference_tangent!(
        second_piola_kirchhoff,
        second_piola_kirchhoff_stress,
        second_piola_kirchhoff_tangent_stiffness,
        SecondPiolaKirchhoffTangentStiffness
    );
}

mod lagrangian {
    use super::{deformation_gradient, deformation_gradient_rate, *};
    use crate::{
        constitutive::{
            fluid::hyperviscous::SaintVenantKirchhoff as Viscous,
            solid::hyperelastic::SaintVenantKirchhoff as Elastic,
        },
        mechanics::{
            CauchyTangentStiffness, FirstPiolaKirchhoffTangentStiffness,
            SecondPiolaKirchhoffTangentStiffness,
        },
    };
    fn model() -> Canonical<Elastic, Viscous> {
        (
            Elastic {
                bulk_modulus: Stress::pascals(13.0),
                shear_modulus: Stress::pascals(3.0),
            },
            Viscous {
                bulk_viscosity: Viscosity::pascal_seconds(11.0),
                shear_viscosity: Viscosity::pascal_seconds(1.0),
            },
        )
            .into()
    }
    crate::constitutive::fluid::viscous::test::finite_difference_tangent!(
        cauchy,
        cauchy_stress,
        cauchy_tangent_stiffness,
        CauchyTangentStiffness
    );
    crate::constitutive::fluid::viscous::test::finite_difference_tangent!(
        first_piola_kirchhoff,
        first_piola_kirchhoff_stress,
        first_piola_kirchhoff_tangent_stiffness,
        FirstPiolaKirchhoffTangentStiffness
    );
    crate::constitutive::fluid::viscous::test::finite_difference_tangent!(
        second_piola_kirchhoff,
        second_piola_kirchhoff_stress,
        second_piola_kirchhoff_tangent_stiffness,
        SecondPiolaKirchhoffTangentStiffness
    );
}
