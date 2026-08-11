use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        test::{rosenbrock, rosenbrock_derivative},
    },
    EqualityConstraint, FirstOrderOptimization, GradientDescent, Scalar, ZerothOrderRootFinding,
};
use crate::math::assert::Assert;
use crate::math::{Current, Dimensionless};

mod minimize {
    use super::*;
    // A scalar unknown needs `Quantity<U>` to be a tensor, since `TensorRank0` is
    // a bare `f64` that cannot carry a unit. Re-enable when it is one.
    #[cfg(any())]
    #[test]
    fn quadratic() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&GradientDescent::default().minimize(
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            1.0,
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &FirstOrderOptimization::<
                Scalar,
                TensorRank1<2, Current>,
                Dimensionless,
                TensorRank1<2, Current>,
            >::minimize(
                &GradientDescent::default(),
                rosenbrock,
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}

mod root {
    use super::*;
    // A scalar unknown needs `Quantity<U>` to be a tensor, since `TensorRank0` is
    // a bare `f64` that cannot carry a unit. Re-enable when it is one.
    #[cfg(any())]
    #[test]
    fn linear() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&GradientDescent::default().root(
            |x: &Scalar| Ok(*x),
            1.0,
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &ZerothOrderRootFinding::<
                TensorRank1<2, Current>,
                Dimensionless,
                TensorRank1<2, Current>,
            >::root(
                &GradientDescent::default(),
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}
