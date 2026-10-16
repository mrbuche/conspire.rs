use super::{super::test::DIFFUSIVITY, Fick};
use crate::{
    constitutive::{chemical::diffusion::ChemicalDiffusion, diffusion::Diffusion},
    math::{
        Quantity, Scalar, Tensor, TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
    },
    mechanics::{
        ConcentrationGradient, MolarFlux, MolarFluxTangent, test::get_concentration_gradient,
    },
};

fn model() -> Fick {
    Fick {
        diffusivity: DIFFUSIVITY,
    }
}

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<Fick>(), std::mem::size_of::<Scalar>())
}

#[test]
fn diffusivity() -> Result<(), AssertionError> {
    model()
        .molar_flux(&get_concentration_gradient())?
        .iter()
        .zip(get_concentration_gradient().iter())
        .try_for_each(|(flux_i, field_i)| Assert::eq(flux_i.value(), &(-3.0 * field_i.value())))
}

#[test]
fn zero() -> Result<(), AssertionError> {
    model()
        .molar_flux(&ConcentrationGradient::from([0.0, 0.0, 0.0]))?
        .iter()
        .try_for_each(Assert::zero)
}

#[test]
fn potential() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().potential(&get_concentration_gradient())?.value(),
        &(0.5 * 3.0 * (12.3 * 12.3 + 5.0 * 5.0 + 8.8 * 8.8)),
    )
}

#[test]
fn tangent() -> Result<(), AssertionError> {
    let tangent = model().molar_flux_tangent(&get_concentration_gradient())?;
    (0..3).try_for_each(|i| {
        (0..3)
            .try_for_each(|j| Assert::eq(tangent[i][j].value(), &(if i == j { -3.0 } else { 0.0 })))
    })
}

#[test]
fn finite_difference_molar_flux() -> Result<(), AssertionError> {
    let concentration_gradient = get_concentration_gradient();
    let flux = model().molar_flux(&concentration_gradient)?;
    let mut fd = MolarFlux::zero();
    for i in 0..3 {
        let mut concentration_gradient_plus = concentration_gradient.clone();
        concentration_gradient_plus[i] += perturbation(0.5 * crate::EPSILON);
        let mut concentration_gradient_minus = concentration_gradient.clone();
        concentration_gradient_minus[i] -= perturbation(0.5 * crate::EPSILON);
        fd[i] = Quantity::new(
            -(model().potential(&concentration_gradient_plus)?.value()
                - model().potential(&concentration_gradient_minus)?.value())
                / crate::EPSILON,
        );
    }
    if flux.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&flux, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_tangent() -> Result<(), AssertionError> {
    let concentration_gradient = get_concentration_gradient();
    let tangent = model().molar_flux_tangent(&concentration_gradient)?;
    let mut fd = MolarFluxTangent::zero();
    for j in 0..3 {
        let mut concentration_gradient_plus = concentration_gradient.clone();
        concentration_gradient_plus[j] += perturbation(0.5 * crate::EPSILON);
        let flux_plus = model().molar_flux(&concentration_gradient_plus)?;
        let mut concentration_gradient_minus = concentration_gradient.clone();
        concentration_gradient_minus[j] -= perturbation(0.5 * crate::EPSILON);
        let flux_minus = model().molar_flux(&concentration_gradient_minus)?;
        for i in 0..3 {
            fd[i][j] = (flux_plus[i] - flux_minus[i])
                / perturbation::<crate::units::ConcentrationPerLength>(crate::EPSILON);
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent, &fd)
    } else {
        Ok(())
    }
}
