use super::*;
use crate::{
    math::{
        TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
    },
    mechanics::SecondPiolaKirchhoffStress,
    units::Rate,
};

const BULK_VISCOSITY: Quantity<Viscosity> = Viscosity::pascal_seconds(11.0);
const SHEAR_VISCOSITY: Quantity<Viscosity> = Viscosity::pascal_seconds(1.0);

fn model() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        bulk_viscosity: BULK_VISCOSITY,
        shear_viscosity: SHEAR_VISCOSITY,
    }
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
fn zero_rate() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        &model().viscous_second_piola_kirchhoff_stress(
            &deformation_gradient(),
            &DeformationGradientRate::zero(),
        )?,
        &SecondPiolaKirchhoffStress::zero(),
    )
}

#[test]
fn finite_difference() -> Result<(), AssertionError> {
    let deformation_gradient = deformation_gradient();
    let deformation_gradient_rate = deformation_gradient_rate();
    let model = model();
    let tangent = model.viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &deformation_gradient,
        &deformation_gradient_rate,
    )?;
    let mut fd = SecondPiolaKirchhoffRateTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut rate_plus = deformation_gradient_rate.clone();
            rate_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let stress_plus =
                model.viscous_second_piola_kirchhoff_stress(&deformation_gradient, &rate_plus)?;
            let mut rate_minus = deformation_gradient_rate.clone();
            rate_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let stress_minus =
                model.viscous_second_piola_kirchhoff_stress(&deformation_gradient, &rate_minus)?;
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
