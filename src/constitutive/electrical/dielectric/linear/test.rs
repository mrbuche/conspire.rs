use super::{super::test::PERMITTIVITY, Linear};
use crate::{
    constitutive::{diffusion::Diffusion, electrical::dielectric::Dielectric},
    math::{
        Quantity, Scalar, Tensor, TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
    },
    mechanics::{ElectricDisplacement, ElectricField, ElectricTangent, test::get_electric_field},
    units::Stress,
};

fn model() -> Linear {
    Linear {
        permittivity: PERMITTIVITY,
    }
}

#[test]
fn size() {
    assert_eq!(std::mem::size_of::<Linear>(), std::mem::size_of::<Scalar>())
}

#[test]
fn permittivity() -> Result<(), AssertionError> {
    model()
        .electric_displacement(&get_electric_field())?
        .iter()
        .zip(get_electric_field().iter())
        .try_for_each(|(displacement_i, field_i)| {
            Assert::eq(displacement_i.value(), &(-3.0 * field_i.value()))
        })
}

#[test]
fn zero() -> Result<(), AssertionError> {
    model()
        .electric_displacement(&ElectricField::from([0.0, 0.0, 0.0]))?
        .iter()
        .try_for_each(Assert::zero)
}

#[test]
fn potential() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        model().potential(&get_electric_field())?,
        &Stress::pascals(0.5 * 3.0 * (12.3 * 12.3 + 5.0 * 5.0 + 8.8 * 8.8)),
    )
}

#[test]
fn tangent() -> Result<(), AssertionError> {
    let tangent = model().electric_tangent(&get_electric_field())?;
    (0..3).try_for_each(|i| {
        (0..3)
            .try_for_each(|j| Assert::eq(tangent[i][j].value(), &(if i == j { -3.0 } else { 0.0 })))
    })
}

#[test]
fn finite_difference_displacement() -> Result<(), AssertionError> {
    let electric_field = get_electric_field();
    let displacement = model().electric_displacement(&electric_field)?;
    let mut fd = ElectricDisplacement::zero();
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
    if displacement.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&displacement, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn finite_difference_tangent() -> Result<(), AssertionError> {
    let electric_field = get_electric_field();
    let tangent = model().electric_tangent(&electric_field)?;
    let mut fd = ElectricTangent::zero();
    for j in 0..3 {
        let mut electric_field_plus = electric_field.clone();
        electric_field_plus[j] += perturbation(0.5 * crate::EPSILON);
        let displacement_plus = model().electric_displacement(&electric_field_plus)?;
        let mut electric_field_minus = electric_field.clone();
        electric_field_minus[j] -= perturbation(0.5 * crate::EPSILON);
        let displacement_minus = model().electric_displacement(&electric_field_minus)?;
        for i in 0..3 {
            fd[i][j] = (displacement_plus[i] - displacement_minus[i])
                / perturbation::<crate::units::ElectricField>(crate::EPSILON);
        }
    }
    if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
        Assert::default().eq_within_fd_tol(&tangent, &fd)
    } else {
        Ok(())
    }
}
