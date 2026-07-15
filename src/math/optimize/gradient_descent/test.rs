use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        test::{rosenbrock, rosenbrock_derivative},
    },
    EqualityConstraint, FirstOrderOptimization, GradientDescent, Scalar, ZerothOrderRootFinding,
};
use crate::math::assert::Assert;

mod minimize {
    use super::*;
    #[test]
    fn quadratic() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            GradientDescent::default().minimize(
                |x: &Scalar| Ok(x.powi(2) / 2.0),
                |x: &Scalar| Ok(*x),
                1.0,
                EqualityConstraint::None,
            )?,
            &0.0,
        )
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &GradientDescent::default().minimize(
                rosenbrock,
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, 1>::identity(),
        )
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            GradientDescent::default().root(|x: &Scalar| Ok(*x), 1.0, EqualityConstraint::None)?,
            &0.0,
        )
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &GradientDescent::default().root(
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, 1>::identity(),
        )
    }
}
