use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        test::{rosenbrock, rosenbrock_derivative},
    },
    EqualityConstraint, FirstOrderOptimization, GradientDescent, Scalar, ZerothOrderRootFinding,
};
use crate::math::assert::Assert;
use crate::math::{Current, Dimensionless, Quantity};

mod minimize {
    use super::*;
    // A scalar unknown is a `Quantity`, a bare `f64` carrying no unit for the
    // step size to be measured against.
    #[test]
    fn quadratic() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&FirstOrderOptimization::<
            Scalar,
            Quantity,
            Dimensionless,
            Quantity,
        >::minimize(
            &GradientDescent::default(),
            |x: &Quantity| Ok(x.powi(2).value() / 2.0),
            |x: &Quantity| Ok(*x),
            Quantity::new(1.0),
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
    #[test]
    fn linear() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&ZerothOrderRootFinding::<
            Quantity,
            Dimensionless,
            Quantity,
        >::root(
            &GradientDescent::default(),
            |x: &Quantity| Ok(*x),
            Quantity::new(1.0),
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
