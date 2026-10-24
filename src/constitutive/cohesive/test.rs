use super::StiffnessCohesive;
use crate::{
    EPSILON,
    math::{
        CrossProduct, TensorArray,
        assert::{Assert, AssertionError, perturbation},
    },
    mechanics::{Normal, Scalar, Separation, Traction},
    units::Length,
};

pub fn perpendiculars(normal: &Normal) -> (Normal, Normal) {
    let first = (Normal::from([1.0, 0.0, 0.0]) - normal * normal[0]).normalized();
    let second = normal.cross(&first);
    (first, second)
}

pub fn check_stiffness(
    traction: impl Fn(Separation, Normal) -> Traction,
    stiffness: &StiffnessCohesive,
    separation: &Separation,
    normal: &Normal,
) -> Result<(), AssertionError> {
    for j in 0..3 {
        let mut step = Separation::zero();
        step[j] = perturbation(0.5 * EPSILON);
        let numerical = (traction(separation + &step, normal.clone())
            - traction(separation - &step, normal.clone()))
            / perturbation::<Length>(EPSILON);
        for i in 0..3 {
            Assert::default().eq_within_fd_tol(numerical[i], &stiffness.0[i][j])?;
        }
    }
    let (first, second) = perpendiculars(normal);
    for direction in [first, second] {
        let step = &direction * (0.5 * EPSILON);
        let rotated = |sign: Scalar| (normal + &step * sign).normalized();
        let numerical = (traction(separation.clone(), rotated(1.0))
            - traction(separation.clone(), rotated(-1.0)))
            / EPSILON;
        let analytical = &stiffness.1 * &direction;
        for i in 0..3 {
            Assert::default().eq_within_fd_tol(numerical[i], &analytical[i])?;
        }
    }
    Ok(())
}
