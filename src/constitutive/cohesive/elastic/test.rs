use super::{Elastic, LinearElastic};
use crate::{
    constitutive::cohesive::test::check_stiffness,
    math::assert::AssertionError,
    mechanics::{Normal, Separation},
    units::{Length, StressPerLength},
};

fn model() -> LinearElastic {
    LinearElastic {
        normal_stiffness: StressPerLength::pascals_per_meter(3.4),
        tangential_stiffness: StressPerLength::pascals_per_meter(1.3),
    }
}

fn check(separation: Separation, normal: Normal) -> Result<(), AssertionError> {
    let stiffness = model()
        .stiffness(separation.clone(), normal.clone())
        .unwrap();
    check_stiffness(
        |separation, normal| model().traction(separation, normal).unwrap(),
        &stiffness,
        &separation,
        &normal,
    )
}

fn normals() -> [Normal; 2] {
    [
        Normal::from([0.0, 0.0, 1.0]),
        Normal::from([0.3, -0.5, 0.8]).normalized(),
    ]
}

#[test]
fn stiffness_is_the_derivative_of_the_traction() -> Result<(), AssertionError> {
    for normal in normals() {
        for separation in [[0.1, 0.2, 0.3], [-0.2, 0.05, 0.4], [0.3, -0.1, -0.2]] {
            check(Separation::from(separation), normal.clone())?;
        }
    }
    Ok(())
}

#[test]
fn stiffness_without_tangential_separation() -> Result<(), AssertionError> {
    for normal in normals() {
        for gap in [0.2, -0.3] {
            check(&normal * Length::meters(gap), normal.clone())?;
        }
    }
    Ok(())
}
