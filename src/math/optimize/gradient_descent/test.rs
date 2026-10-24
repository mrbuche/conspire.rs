use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        test::{rosenbrock, rosenbrock_derivative},
    },
    BarzilaiBorwein, EqualityConstraint, GradientDescent, Optimization, RootFinding, Tolerances,
};
use crate::math::{Current, Matrix, Quantity, Vector};
use crate::math::{assert::Assert, optimize::BarzilaiBorweinStep};

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

mod step_estimator {
    use super::*;
    fn minimized(step: BarzilaiBorweinStep) -> Result<Vector, super::super::OptimizationError> {
        GradientDescent {
            max_steps: 1000,
            ..GradientDescent::default()
        }
        .with_step_estimator(BarzilaiBorwein { step })
        .minimize(
            |x: &Vector| Ok((x[0].powi(2) + 10.0 * x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(Vector::from([x[0], 10.0 * x[1]])),
            |_| Ok(()),
            Vector::from([3.0, -2.0]),
            vec![EqualityConstraint::None],
            None,
        )
    }
    #[test]
    fn short() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(BarzilaiBorweinStep::Short)?)
    }
    #[test]
    fn long() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(BarzilaiBorweinStep::Long)?)
    }
    #[test]
    fn alternating() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(BarzilaiBorweinStep::Alternating)?)
    }
    #[test]
    fn the_default_is_the_short_step() {
        assert!(format!("{:?}", GradientDescent::default().step_estimator).contains("Short"))
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

mod linear_constraint {
    use super::*;
    fn minimized(dual: bool) -> Result<Vector, super::super::OptimizationError> {
        let mut matrix = Matrix::zero(1, 2);
        matrix[0][0] = 1.0;
        matrix[0][1] = 1.0;
        GradientDescent {
            abs_tol: Tolerances {
                constraint: 1e-6,
                residual: 1e-6,
            },
            dual,
            max_steps: 5000,
            ..GradientDescent::default()
        }
        .minimize(
            |x: &Vector| Ok((x[0].powi(2) + x[1].powi(2)) / 2.0),
            |x: &Vector| Ok(x.clone()),
            |_| Ok(()),
            Vector::from([4.0, -3.0]),
            vec![EqualityConstraint::Linear(matrix, Vector::from([2.0]))],
            None,
        )
    }
    #[test]
    fn dual() -> Result<(), AssertionError> {
        Assert {
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            ..Default::default()
        }
        .eq_within_tols(&minimized(true)?, &Vector::from([1.0, 1.0]))
    }
}
