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
            vec![EqualityConstraint::None],
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
                vec![EqualityConstraint::None],
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
            vec![EqualityConstraint::None],
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
                vec![EqualityConstraint::None],
                None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}

mod prescribed {
    use super::*;
    use crate::math::optimize::Prescription;
    type Point = TensorRank1<2, Current>;
    fn target() -> Point {
        TensorRank1::from([4.0, 7.0])
    }
    fn root(prescribed: Vec<(usize, Prescription)>) -> Result<Point, AssertionError> {
        Ok(GradientDescent::default().root(
            |x: &Point| Ok(x - &target()),
            |_| Ok(()),
            TensorRank1::from([0.0, 3.0]),
            vec![EqualityConstraint::Prescribed(prescribed)],
            None,
        )?)
    }
    fn minimum(prescribed: Vec<(usize, Prescription)>) -> Result<Point, AssertionError> {
        Ok(GradientDescent::default().minimize(
            |x: &Point| Ok(0.5 * ((x[0].value() - 4.0).powi(2) + (x[1].value() - 7.0).powi(2))),
            |x: &Point| Ok(x - &target()),
            |_| Ok(()),
            TensorRank1::from([0.0, 3.0]),
            vec![EqualityConstraint::Prescribed(prescribed)],
            None,
        )?)
    }
    #[test]
    fn an_absolute_value() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &root(vec![(1, Prescription::Absolute(5.0))])?,
            &Point::from([4.0, 5.0]),
        )
    }
    #[test]
    fn an_offset() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &root(vec![(1, Prescription::Offset(0.5))])?,
            &Point::from([4.0, 3.5]),
        )
    }
    #[test]
    fn a_hold() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &root(vec![(1, Prescription::Initial)])?,
            &Point::from([4.0, 3.0]),
        )
    }
    #[test]
    fn minimizing_an_absolute_value() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimum(vec![(1, Prescription::Absolute(5.0))])?,
            &Point::from([4.0, 5.0]),
        )
    }
    #[test]
    fn minimizing_an_offset() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimum(vec![(0, Prescription::Offset(-1.0))])?,
            &Point::from([-1.0, 7.0]),
        )
    }
}
