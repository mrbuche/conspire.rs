use super::{
    EqualityConstraint, FirstOrderOptimization, GradientDescent, Scalar, ZerothOrderRootFinding,
};

const TOLERANCE: Scalar = 1e-5;

mod minimize {
    use super::*;
    #[test]
    fn quadratic() {
        let x = GradientDescent {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            1.0,
            EqualityConstraint::None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
    #[test]
    fn cubic() {
        let x = GradientDescent {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(x.powi(3) / 6.0),
            |x: &Scalar| Ok(x.powi(2) / 2.0),
            1.0,
            EqualityConstraint::None,
        )
        .unwrap();
        assert!(x.abs() < TOLERANCE)
    }
    #[test]
    fn sin() {
        let x = GradientDescent {
            ..Default::default()
        }
        .minimize(
            |x: &Scalar| Ok(-x.sin()),
            |x: &Scalar| Ok(x.sin()),
            1.0,
            EqualityConstraint::None,
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
            GradientDescent {
                ..Default::default()
            }
            .root(|x: &Scalar| Ok(*x), 1.0, EqualityConstraint::None,)
            .unwrap()
            .abs()
                < TOLERANCE
        )
    }
    #[test]
    fn quadratic() {
        assert!(
            GradientDescent {
                ..Default::default()
            }
            .root(
                |x: &Scalar| Ok(x.powi(2) / 2.0),
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
            GradientDescent {
                ..Default::default()
            }
            .root(|x: &Scalar| Ok(x.sin()), 1.0, EqualityConstraint::None,)
            .unwrap()
            .abs()
                < TOLERANCE
        )
    }
}
