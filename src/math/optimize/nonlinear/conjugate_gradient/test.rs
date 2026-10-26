use crate::math::{
    Current, Matrix, Tensor, TensorRank1, Vector,
    assert::{Assert, AssertionError},
    optimize::{
        Armijo, Conjugacy, ConjugateGradient, EqualityConstraint, Optimization, OptimizationError,
        Tolerances, Wolfe,
        test::{rosenbrock, rosenbrock_derivative},
    },
};

fn quadratic(x: &Vector) -> Result<f64, String> {
    Ok((x[0].powi(2) + 100.0 * x[1].powi(2)) / 2.0)
}

fn quadratic_derivative(x: &Vector) -> Result<Vector, String> {
    Ok(Vector::from([x[0], 100.0 * x[1]]))
}

fn minimized<S, Q>(
    solver: &ConjugateGradient<S, Q>,
    constraint: EqualityConstraint,
) -> Result<Vector, OptimizationError>
where
    ConjugateGradient<S, Q>: Optimization<f64, Vector, (), Vector>,
{
    solver.minimize(
        quadratic,
        quadratic_derivative,
        |_| Ok(()),
        Vector::from([3.0, -2.0]),
        vec![constraint],
        None,
    )
}

fn near_one(x: &Vector) -> bool {
    x.iter().all(|entry| (entry - 1.0).abs() < 1e-8)
}

fn rosenbrock_with(conjugacy: Conjugacy, max_steps: usize) -> Result<Vector, OptimizationError> {
    ConjugateGradient {
        conjugacy,
        max_steps,
        abs_tol: Tolerances {
            residual: 1e-10,
            ..Tolerances::default()
        },
        ..ConjugateGradient::default()
    }
    .minimize(
        rosenbrock,
        rosenbrock_derivative,
        |_| Ok(()),
        Vector::from([-1.0, 1.0]),
        vec![EqualityConstraint::None],
        None,
    )
}

mod minimize {
    use super::*;
    #[test]
    fn quadratic_anisotropic() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(
            &ConjugateGradient::default(),
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn quadratic_with_wolfe() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(
            &ConjugateGradient::default().with_line_search(Wolfe {
                control_1: 1e-4,
                control_2: 0.999,
                cut_back: 0.5,
                max_steps: 100,
                strong: true,
            }),
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn quadratic_with_armijo() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(
            &ConjugateGradient::default().with_line_search(Armijo::default()),
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn rosenbrock_tensor() -> Result<(), OptimizationError> {
        let x = ConjugateGradient {
            abs_tol: Tolerances {
                residual: 1e-10,
                ..Tolerances::default()
            },
            ..ConjugateGradient::default()
        }
        .minimize(
            rosenbrock,
            rosenbrock_derivative,
            |_| Ok(()),
            TensorRank1::<2, Current>::from([-1.0, 1.0]),
            vec![EqualityConstraint::None],
            None,
        )?;
        assert!(x.iter().all(|entry| (entry.value() - 1.0).abs() < 1e-8));
        Ok(())
    }
    #[test]
    fn fixed_entries_stay_put() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized(
                &ConjugateGradient::default(),
                EqualityConstraint::Fixed(vec![0]),
            )?,
            &Vector::from([3.0, 0.0]),
        )
    }
    #[test]
    fn linear_constraints_are_refused() {
        assert!(
            minimized(
                &ConjugateGradient::default(),
                EqualityConstraint::Linear(Matrix::zero(1, 2), Vector::zero(1)),
            )
            .is_err()
        )
    }
    #[test]
    fn relative_tolerance_stops_sooner() {
        let solver = |rel_tol| ConjugateGradient {
            max_steps: 1,
            rel_tol,
            ..ConjugateGradient::default()
        };
        assert!(minimized(&solver(None), EqualityConstraint::None).is_err());
        assert!(minimized(&solver(Some(0.5)), EqualityConstraint::None).is_ok());
    }
    #[test]
    fn quadratic_in_three_steps() -> Result<(), AssertionError> {
        Assert::default().zero_within_tols(&minimized(
            &ConjugateGradient {
                max_steps: 3,
                ..ConjugateGradient::default()
            },
            EqualityConstraint::None,
        )?)
    }
    #[test]
    fn a_loose_absolute_tolerance_ends_the_descent_where_it_is() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            &minimized(
                &ConjugateGradient {
                    abs_tol: Tolerances {
                        residual: 1e3,
                        ..Tolerances::default()
                    },
                    ..ConjugateGradient::default()
                },
                EqualityConstraint::None,
            )?,
            &Vector::from([3.0, -2.0]),
        )
    }
    #[test]
    fn maximum_steps() {
        assert!(matches!(
            rosenbrock_with(Conjugacy::default(), 3),
            Err(OptimizationError::MaximumStepsReached(3, _))
        ));
    }
}

mod weight {
    use super::super::restart_if_unbounded;
    use crate::{math::Quantity, units::Dimensionless};
    fn bounded(weight: f64) -> f64 {
        restart_if_unbounded(Quantity::<Dimensionless>::new(weight)).value()
    }
    #[test]
    fn a_positive_weight_is_kept() {
        assert_eq!(bounded(0.5), 0.5)
    }
    #[test]
    fn a_negative_weight_restarts() {
        assert_eq!(bounded(-0.5), 0.0)
    }
    #[test]
    fn an_infinite_weight_restarts() {
        assert_eq!(bounded(f64::INFINITY), 0.0)
    }
    #[test]
    fn an_undefined_weight_restarts() {
        assert_eq!(bounded(f64::NAN), 0.0)
    }
}

mod units {
    use super::*;
    use crate::{
        math::Quantity,
        units::{Dimensionless, Length, ReciprocalLength},
    };
    #[test]
    fn a_quantity_with_a_unit_is_minimized() -> Result<(), OptimizationError> {
        let x = ConjugateGradient::default().minimize(
            |x: &Quantity<Length>| {
                Ok(Quantity::<Dimensionless>::new(
                    (x.value() - 1.0).powi(2) / 2.0,
                ))
            },
            |x: &Quantity<Length>| Ok(Quantity::<ReciprocalLength>::new(x.value() - 1.0)),
            |_| Ok(()),
            Quantity::<Length>::new(4.0),
            vec![EqualityConstraint::None],
            None,
        )?;
        assert!((x.value() - 1.0).abs() < 1e-6);
        Ok(())
    }
}

mod conjugacy {
    use super::*;
    fn solves(conjugacy: Conjugacy, max_steps: usize) -> Result<(), OptimizationError> {
        assert!(near_one(&rosenbrock_with(conjugacy, max_steps)?));
        Ok(())
    }
    #[test]
    fn polak_ribiere() -> Result<(), OptimizationError> {
        solves(Conjugacy::PolakRibiere, 120)
    }
    #[test]
    fn hestenes_stiefel() -> Result<(), OptimizationError> {
        solves(Conjugacy::HestenesStiefel, 175)
    }
    #[test]
    fn fletcher_reeves_stalls_where_the_others_do_not() -> Result<(), OptimizationError> {
        assert!(rosenbrock_with(Conjugacy::FletcherReeves, 250).is_err());
        solves(Conjugacy::FletcherReeves, 5000)
    }
}
