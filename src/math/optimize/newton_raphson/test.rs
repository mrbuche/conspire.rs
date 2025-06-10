use super::{
    EqualityConstraint, FirstOrderRootFinding, NewtonRaphson, Scalar, SecondOrderOptimization, LineSearch,
};

const TOLERANCE: Scalar = 1e-6;

mod minimize {
    use super::*;
    #[test]
    fn armijo() {
        let x = NewtonRaphson {
            line_search: Some(LineSearch::default()),
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(x.powi(3) / 6.0),
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            1.0,
            EqualityConstraint::None,
            None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
    #[test]
    fn quadratic() {
        let x = NewtonRaphson {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            |_: &Scalar| Ok(1.0),
            1.0,
            EqualityConstraint::None,
            None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
    #[test]
    fn cubic() {
        let x = NewtonRaphson {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(x.powi(3) / 6.0),
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            1.0,
            EqualityConstraint::None,
            None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
    #[test]
    fn sin() {
        let x = NewtonRaphson {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(-x.sin()),
            |x: &Scalar| Ok(x.sin()),
            |x: &Scalar| Ok(x.cos()),
            1.0,
            EqualityConstraint::None,
            None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() {
        assert!(
            NewtonRaphson {
                ..Default::default()
            }
            .root(
                |x: &Scalar| Ok(*x),
                |_: &Scalar| Ok(1.0),
                1.0,
                EqualityConstraint::None,
            )
            .unwrap()
            .abs()
                < TOLERANCE
        )
    }
    #[test]
    fn quadratic() {
        assert!(
            NewtonRaphson {
                ..Default::default()
            }
            .root(
                |x: &Scalar| Ok(x.powi(2) / 2.0),
                |x: &Scalar| Ok(*x),
                1.0,
                EqualityConstraint::None,
            )
            .unwrap()
            .abs()
                < TOLERANCE
        )
    }
    #[test]
    fn sin() {
        assert!(
            NewtonRaphson {
                ..Default::default()
            }
            .root(
                |x: &Scalar| Ok(x.sin()),
                |x: &Scalar| Ok(x.cos()),
                1.0,
                EqualityConstraint::None,
            )
            .unwrap()
            .abs()
                < TOLERANCE
        )
    }
}
