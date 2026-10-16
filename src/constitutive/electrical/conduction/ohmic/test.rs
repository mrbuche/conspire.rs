use super::{super::test::ELECTRICAL_CONDUCTIVITY, Ohmic};
use crate::{
    constitutive::{diffusion::Diffusion, electrical::conduction::ElectricalConduction},
    math::{
        Quantity, Scalar, Tensor, TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
    },
    mechanics::{CurrentDensity, CurrentDensityTangent, ElectricField, test::get_electric_field},
    units::{Rate, Stress},
};

fn model() -> Ohmic {
    Ohmic {
        electrical_conductivity: ELECTRICAL_CONDUCTIVITY,
    }
}

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<Ohmic>(), std::mem::size_of::<Scalar>())
}

#[test]
fn electrical_conductivity() -> Result<(), AssertionError> {
    model()
        .current_density(&get_electric_field())?
        .iter()
        .zip(get_electric_field().iter())
        .try_for_each(|(current_i, field_i)| {
            Assert::eq(current_i.value(), &(-3.0 * field_i.value()))
        })
}

#[test]
fn zero() -> Result<(), AssertionError> {
    model()
        .current_density(&ElectricField::from([0.0, 0.0, 0.0]))?
        .iter()
        .try_for_each(Assert::zero)
}

#[test]
fn potential() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().potential(&get_electric_field())?,
        &(Stress::pascals(0.5 * 3.0 * (12.3 * 12.3 + 5.0 * 5.0 + 8.8 * 8.8))
            * Rate::per_second(1.0)),
    )
}

#[test]
fn tangent() -> Result<(), AssertionError> {
    let tangent = model().current_density_tangent(&get_electric_field())?;
    (0..3).try_for_each(|i| {
        (0..3)
            .try_for_each(|j| Assert::eq(tangent[i][j].value(), &(if i == j { -3.0 } else { 0.0 })))
    })
}

#[test]
fn finite_difference_current_density() -> Result<(), AssertionError> {
    let electric_field = get_electric_field();
    let current = model().current_density(&electric_field)?;
    let mut fd = CurrentDensity::zero();
    for i in 0..3 {
        let mut electric_field_plus = electric_field.clone();
        electric_field_plus[i] += perturbation(0.5 * crate::EPSILON);
        let mut electric_field_minus = electric_field.clone();
        electric_field_minus[i] -= perturbation(0.5 * crate::EPSILON);
        fd[i] = Quantity::new(
            -(model().potential(&electric_field_plus)?.value()
                - model().potential(&electric_field_minus)?.value())
                / crate::EPSILON,
        );
    }
    if current.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&current, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_tangent() -> Result<(), AssertionError> {
    let electric_field = get_electric_field();
    let tangent = model().current_density_tangent(&electric_field)?;
    let mut fd = CurrentDensityTangent::zero();
    for j in 0..3 {
        let mut electric_field_plus = electric_field.clone();
        electric_field_plus[j] += perturbation(0.5 * crate::EPSILON);
        let current_plus = model().current_density(&electric_field_plus)?;
        let mut electric_field_minus = electric_field.clone();
        electric_field_minus[j] -= perturbation(0.5 * crate::EPSILON);
        let current_minus = model().current_density(&electric_field_minus)?;
        for i in 0..3 {
            fd[i][j] = (current_plus[i] - current_minus[i])
                / perturbation::<crate::units::ElectricField>(crate::EPSILON);
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent, &fd)
    } else {
        Ok(())
    }
}
