use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        test::{rosenbrock, rosenbrock_derivative},
    },
    EqualityConstraint, GradientDescent, Optimization, RootFinding,
};
use crate::math::assert::Assert;
use crate::math::{Current, Quantity};

mod minimize {
    use super::*;
    #[test]
    fn quadratic() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&GradientDescent::default().minimize(
            |x: &Quantity| Ok(x.powi(2).value() / 2.0),
            |x: &Quantity| Ok(*x),
            |_| Ok(()),
            Quantity::new(1.0),
            EqualityConstraint::None,
            None,
        )?)
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &GradientDescent::default().minimize(
                rosenbrock,
                rosenbrock_derivative,
                |_| Ok(()),
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
                None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&GradientDescent::default().root(
            |x: &Quantity| Ok(*x),
            |_| Ok(()),
            Quantity::new(1.0),
            EqualityConstraint::None,
            None,
        )?)
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &GradientDescent::default().root(
                rosenbrock_derivative,
                |_| Ok(()),
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
                None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}
