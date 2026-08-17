use super::{
    super::{
        super::{TensorArray, TensorRank1, assert::AssertionError},
        LineSearch,
        test::{rosenbrock, rosenbrock_derivative},
    },
    Conjugacy, ConjugateGradient, EqualityConstraint, FirstOrderOptimization,
    ZerothOrderRootFinding,
};
use crate::math::assert::Assert;
use crate::math::{Current, Matrix, Quantity, Scalar, Vector};

const CONJUGACIES: [Conjugacy; 3] = [
    Conjugacy::FletcherReeves,
    Conjugacy::HestenesStiefel,
    Conjugacy::PolakRibiere,
];

/// What the argument is worth once the residual has met its own tolerance.
///
/// The argument is only as accurate as the residual tolerance divided by the
/// curvature there, so a steep problem converged on its gradient lands a little
/// further out than that gradient suggests.
const ASSERT: Assert = Assert {
    abs_tol: 1e-10,
    fd_tol: 1e-10,
    rel_tol: 1e-10,
};

fn solver(conjugacy: Conjugacy) -> ConjugateGradient {
    ConjugateGradient {
        conjugacy,
        ..Default::default()
    }
}

/// An anisotropic quadratic, whose conditioning is what conjugacy is for.
fn quadratic(x: &TensorRank1<2, Current>) -> Result<Scalar, String> {
    Ok((x[0].value().powi(2) * 100.0 + x[1].value().powi(2)) / 2.0)
}

fn quadratic_derivative(x: &TensorRank1<2, Current>) -> Result<TensorRank1<2, Current>, String> {
    Ok(TensorRank1::from([x[0].value() * 100.0, x[1].value()]))
}

/// The anisotropic quadratic held to a line, and where that puts its minimum.
///
/// The gradient there is normal to the line, so `100 x = y` alongside `x + y = 1`.
fn line() -> (Matrix, Vector) {
    let mut matrix = Matrix::zero(1, 2);
    matrix[0][0] = 1.0;
    matrix[0][1] = 1.0;
    (matrix, Vector::from([1.0]))
}

mod minimize {
    use super::*;
    #[test]
    fn quadratic_1d() -> Result<(), AssertionError> {
        for conjugacy in CONJUGACIES {
            ASSERT.zero_within_tols(&solver(conjugacy).minimize(
                |x: &Quantity| Ok(x.powi(2).value() / 2.0),
                |x: &Quantity| Ok(*x),
                Quantity::new(1.0),
                EqualityConstraint::None,
            )?)?
        }
        Ok(())
    }
    #[test]
    fn quadratic_2d() -> Result<(), AssertionError> {
        for conjugacy in CONJUGACIES {
            ASSERT.zero_within_tols(&solver(conjugacy).minimize(
                quadratic,
                quadratic_derivative,
                TensorRank1::from([1.0, 1.0]),
                EqualityConstraint::None,
            )?)?
        }
        Ok(())
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        ASSERT.eq_within_tols(
            &solver(Conjugacy::PolakRibiere).minimize(
                rosenbrock,
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
    /// Armijo only ever shortens a step, so it drags where the secant estimate
    /// alone would have kept its length.
    #[test]
    fn rosenbrock_2d_armijo() -> Result<(), AssertionError> {
        ASSERT.eq_within_tols(
            &ConjugateGradient {
                line_search: LineSearch::default(),
                max_steps: 500,
                ..Default::default()
            }
            .minimize(
                rosenbrock,
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
    /// Hestenes-Stiefel divides by the previous direction against the change in
    /// the gradient, which is the very quantity the curvature condition holds
    /// away from zero. It converges only once the line search enforces one.
    #[test]
    fn rosenbrock_2d_hestenes_stiefel_needs_curvature() -> Result<(), AssertionError> {
        let solver = |line_search| ConjugateGradient {
            conjugacy: Conjugacy::HestenesStiefel,
            line_search,
            max_steps: 2500,
            ..Default::default()
        };
        let start = TensorRank1::<2, Current>::from([-1.0, 1.0]);
        assert!(
            solver(LineSearch::None)
                .minimize(
                    rosenbrock,
                    rosenbrock_derivative,
                    start.clone(),
                    EqualityConstraint::None
                )
                .is_err()
        );
        ASSERT.eq_within_tols(
            &solver(LineSearch::Wolfe {
                control_1: 1e-4,
                control_2: 9e-1,
                cut_back: 5e-1,
                max_steps: 100,
                strong: true,
            })
            .minimize(
                rosenbrock,
                rosenbrock_derivative,
                start,
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
    /// Fletcher-Reeves has no clamp to restart it, so it stalls where
    /// Polak-Ribière recovers.
    #[test]
    fn rosenbrock_2d_fletcher_reeves_stalls() {
        assert!(
            solver(Conjugacy::FletcherReeves)
                .minimize(
                    rosenbrock,
                    rosenbrock_derivative,
                    TensorRank1::<2, Current>::from([-1.0, 1.0]),
                    EqualityConstraint::None,
                )
                .is_err()
        )
    }
}

mod constrained {
    use super::*;
    use crate::math::TensorRank1Vec;

    /// The same quadratic over the type a constrained problem is posed in.
    fn quadratic(x: &TensorRank1Vec<2, Current>) -> Result<Scalar, String> {
        Ok((x[0][0].value().powi(2) * 100.0 + x[0][1].value().powi(2)) / 2.0)
    }
    fn quadratic_derivative(
        x: &TensorRank1Vec<2, Current>,
    ) -> Result<TensorRank1Vec<2, Current>, String> {
        Ok(TensorRank1Vec::from([TensorRank1::from([
            x[0][0].value() * 100.0,
            x[0][1].value(),
        ])]))
    }
    fn at(entries: [Scalar; 2]) -> TensorRank1Vec<2, Current> {
        TensorRank1Vec::from([TensorRank1::from(entries)])
    }
    fn on_line() -> TensorRank1Vec<2, Current> {
        at([1.0 / 101.0, 100.0 / 101.0])
    }
    /// Stepping the variables and the multipliers together is slow whatever the
    /// direction, the step being the shorter of what the two blocks ask for.
    #[test]
    fn linear() -> Result<(), AssertionError> {
        let (matrix, rhs) = line();
        ASSERT.eq_within_tols(
            &ConjugateGradient {
                max_steps: 2500,
                ..Default::default()
            }
            .minimize(
                quadratic,
                quadratic_derivative,
                at([1.0, 1.0]),
                EqualityConstraint::Linear(matrix, rhs),
            )?,
            &on_line(),
        )
    }
    #[test]
    fn linear_dual() -> Result<(), AssertionError> {
        let (matrix, rhs) = line();
        ASSERT.eq_within_tols(
            &ConjugateGradient {
                dual: true,
                ..Default::default()
            }
            .minimize(
                quadratic,
                quadratic_derivative,
                at([1.0, 1.0]),
                EqualityConstraint::Linear(matrix, rhs),
            )?,
            &on_line(),
        )
    }
    /// Fixing the first variable leaves the second free to reach its own minimum.
    #[test]
    fn fixed() -> Result<(), AssertionError> {
        ASSERT.eq_within_tols(
            &solver(Conjugacy::PolakRibiere).minimize(
                quadratic,
                quadratic_derivative,
                at([2.0, 1.0]),
                EqualityConstraint::Fixed(vec![0]),
            )?,
            &at([2.0, 0.0]),
        )
    }
}

mod root {
    use super::*;
    #[test]
    fn linear() -> Result<(), AssertionError> {
        for conjugacy in CONJUGACIES {
            ASSERT.zero_within_tols(&solver(conjugacy).root(
                |x: &Quantity| Ok(*x),
                Quantity::new(1.0),
                EqualityConstraint::None,
            )?)?
        }
        Ok(())
    }
    #[test]
    fn quadratic_2d() -> Result<(), AssertionError> {
        for conjugacy in CONJUGACIES {
            ASSERT.zero_within_tols(&solver(conjugacy).root(
                quadratic_derivative,
                TensorRank1::from([1.0, 1.0]),
                EqualityConstraint::None,
            )?)?
        }
        Ok(())
    }
    #[test]
    fn rosenbrock_2d() -> Result<(), AssertionError> {
        ASSERT.eq_within_tols(
            &solver(Conjugacy::PolakRibiere).root(
                rosenbrock_derivative,
                TensorRank1::from([-1.0, 1.0]),
                EqualityConstraint::None,
            )?,
            &TensorRank1::<2, Current>::identity(),
        )
    }
}

#[test]
fn debug() {
    let _ = format!("{:?}", ConjugateGradient::default());
}
