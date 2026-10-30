use super::{Corrections, LimitedMemoryBFGS};
use crate::{
    math::{
        Matrix, Quantity, Reference, Scalar, Tensor, TensorRank1Vec, Vector,
        assert::{Assert, AssertionError},
        optimize::{
            EqualityConstraint, NoSearch, Optimization, OptimizationError, Tolerances, Wolfe,
        },
    },
    units::{Area, Dimensionless, Length},
};

fn solver() -> LimitedMemoryBFGS {
    LimitedMemoryBFGS {
        max_steps: 500,
        ..Default::default()
    }
}

fn rosenbrock(x: &Vector) -> Result<Quantity, String> {
    Ok(Dimensionless::of(
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
    ))
}

fn rosenbrock_gradient(x: &Vector) -> Result<Vector, String> {
    Ok([
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ]
    .into())
}

fn bowl(x: &Vector) -> Result<Quantity, String> {
    Ok(Dimensionless::of(x.iter().map(|x| (x - 1.0).powi(2)).sum()))
}

fn bowl_gradient(x: &Vector) -> Result<Vector, String> {
    Ok(x.iter().map(|x| 2.0 * (x - 1.0)).collect())
}

#[test]
fn rosenbrock_converges() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        &solver().minimize(
            rosenbrock,
            rosenbrock_gradient,
            |_| Ok(()),
            Vector::from([-1.2, 1.0]),
            Vec::new(),
            None,
        )?,
        &Vector::from([1.0, 1.0]),
    )
}

#[test]
fn converges_on_a_typed_problem_with_units() -> Result<(), AssertionError> {
    let solution = solver().minimize(
        |x: &TensorRank1Vec<3, Reference, Length>| {
            Ok(Quantity::<Area>::new(
                x.as_flat().iter().map(|x| (x - 1.0).powi(2)).sum(),
            ))
        },
        |x: &TensorRank1Vec<3, Reference, Length>| {
            Ok(TensorRank1Vec::<3, Reference, Length>::from(
                x.as_flat()
                    .iter()
                    .map(|x| 2.0 * (x - 1.0))
                    .collect::<Vector>(),
            ))
        },
        |_| Ok(()),
        TensorRank1Vec::<3, Reference, Length>::from([[0.0, 2.0, 5.0], [-3.0, 1.0, 0.5]]),
        Vec::new(),
        None,
    )?;
    Assert::default().eq_within_tols(
        Vector::from(solution.as_flat().to_vec()),
        &Vector::from([1.0; 6]),
    )
}

#[test]
fn fixed_entries_stay_put() -> Result<(), AssertionError> {
    let descent = solver().descend(
        bowl,
        bowl_gradient,
        Vector::from([0.0, 0.0, 0.0]),
        vec![EqualityConstraint::Fixed(vec![1])],
    )?;
    assert!(descent.converged);
    Assert::default().eq_within_tols(&descent.solution, &Vector::from([1.0, 0.0, 1.0]))
}

#[test]
fn linear_constraints_are_refused() {
    assert!(matches!(
        solver().descend(
            bowl,
            bowl_gradient,
            Vector::from([0.0, 0.0]),
            vec![EqualityConstraint::Linear(
                Matrix::zero(2, 2),
                Vector::from([0.0, 0.0])
            )],
        ),
        Err(OptimizationError::Intermediate(_))
    ))
}

#[test]
fn reports_where_it_got_to_when_the_budget_runs_out() -> Result<(), OptimizationError> {
    let solver = LimitedMemoryBFGS {
        max_steps: 3,
        ..Default::default()
    };
    let descent = solver.descend(
        rosenbrock,
        rosenbrock_gradient,
        Vector::from([-1.2, 1.0]),
        Vec::new(),
    )?;
    assert!(!descent.converged);
    assert_eq!(descent.steps, 3);
    assert!(descent.value < descent.initial);
    assert!(matches!(
        solver.minimize(
            rosenbrock,
            rosenbrock_gradient,
            |_| Ok(()),
            Vector::from([-1.2, 1.0]),
            Vec::new(),
            None,
        ),
        Err(OptimizationError::MaximumStepsReached(3, _))
    ));
    Ok(())
}

#[test]
fn stops_when_no_step_decreases_the_function() -> Result<(), OptimizationError> {
    let lying = |_: &Vector| Ok(Vector::from([1.0]));
    let rising = |x: &Vector| Ok(Dimensionless::of(-x[0]));
    let descent = solver().descend(rising, lying, Vector::from([0.0]), Vec::new())?;
    assert!(!descent.converged);
    assert!(descent.steps < 500);
    assert!(matches!(
        solver().minimize(
            rising,
            lying,
            |_| Ok(()),
            Vector::from([0.0]),
            Vec::new(),
            None,
        ),
        Err(OptimizationError::Intermediate(_))
    ));
    Ok(())
}

#[test]
fn starts_converged_at_a_minimum() -> Result<(), OptimizationError> {
    let descent = solver().descend(bowl, bowl_gradient, Vector::from([1.0, 1.0]), Vec::new())?;
    assert!(descent.converged);
    assert_eq!(descent.steps, 0);
    Ok(())
}

#[test]
fn relative_tolerance_is_read_against_the_initial_gradient() -> Result<(), OptimizationError> {
    let solver = LimitedMemoryBFGS {
        abs_tol: Tolerances {
            constraint: 0.0,
            residual: 0.0,
        },
        rel_tol: Some(Dimensionless::of(1.0e-6)),
        ..solver()
    };
    let descent = solver.descend(bowl, bowl_gradient, Vector::from([10.0, -7.0]), Vec::new())?;
    assert!(descent.converged);
    assert!(descent.steps > 0);
    Ok(())
}

#[test]
fn takes_whole_steps_without_a_line_search() -> Result<(), OptimizationError> {
    let solver = solver().with_line_search(NoSearch);
    let descent = solver.descend(bowl, bowl_gradient, Vector::from([3.0, 3.0]), Vec::new())?;
    assert!(descent.converged);
    Ok(())
}

#[test]
fn an_ascent_direction_is_never_followed() -> Result<(), OptimizationError> {
    let solver = LimitedMemoryBFGS {
        max_steps: 5,
        typical: -1.0,
        ..Default::default()
    };
    let descent = solver.descend(bowl, bowl_gradient, Vector::from([3.0, 3.0]), Vec::new())?;
    assert!(!descent.converged);
    assert_eq!(descent.steps, 5);
    assert_eq!(descent.value, descent.initial);
    Ok(())
}

#[test]
fn a_flat_function_gives_no_curvature_to_learn_from() -> Result<(), OptimizationError> {
    let solver = LimitedMemoryBFGS {
        max_steps: 8,
        ..Default::default()
    };
    let descent = solver.descend(
        |x: &Vector| Ok(Dimensionless::of(x.iter().sum::<Scalar>())),
        |x: &Vector| Ok(x.iter().map(|_| 1.0).collect::<Vector>()),
        Vector::from([0.0, 0.0]),
        Vec::new(),
    )?;
    assert!(!descent.converged);
    assert!(descent.value < descent.initial);
    Ok(())
}

#[test]
fn a_failing_function_is_an_error() {
    assert!(matches!(
        solver().descend(
            |_: &Vector| Err::<Quantity, _>("broken".to_string()),
            bowl_gradient,
            Vector::from([0.0]),
            Vec::new(),
        ),
        Err(OptimizationError::Intermediate(_))
    ))
}

#[test]
fn a_short_history_still_converges() -> Result<(), AssertionError> {
    let solver = LimitedMemoryBFGS {
        history: 2,
        ..solver()
    };
    Assert::default().eq_within_tols(
        &solver
            .descend(
                rosenbrock,
                rosenbrock_gradient,
                Vector::from([-1.2, 1.0]),
                Vec::new(),
            )
            .map_err(|error| AssertionError::from(format!("{error}")))?
            .solution,
        &Vector::from([1.0, 1.0]),
    )
}

#[test]
fn descending_from_empty_corrections_is_descending() -> Result<(), OptimizationError> {
    let start = Vector::from([-1.2, 1.0]);
    let plain = solver().descend(rosenbrock, rosenbrock_gradient, start.clone(), Vec::new())?;
    let from = solver().descend_from(
        &mut Corrections::default(),
        rosenbrock,
        rosenbrock_gradient,
        start,
        Vec::new(),
    )?;
    assert_eq!(plain.steps, from.steps);
    assert_eq!(plain.value, from.value);
    assert_eq!(plain.solution.as_slice(), from.solution.as_slice());
    Ok(())
}

#[test]
fn kept_corrections_make_the_next_descent_better() -> Result<(), OptimizationError> {
    let curvature = [1.0, 10.0, 100.0, 1.0e3, 1.0e4];
    let value = |x: &Vector| -> Result<Quantity, String> {
        Ok(Dimensionless::of(
            x.iter()
                .zip(curvature)
                .map(|(x, c)| 0.5 * c * (x - 1.0).powi(2))
                .sum(),
        ))
    };
    let slope = |x: &Vector| -> Result<Vector, String> {
        Ok(x.iter()
            .zip(curvature)
            .map(|(x, c)| c * (x - 1.0))
            .collect())
    };
    let short = LimitedMemoryBFGS {
        max_steps: 3,
        ..solver()
    };
    let mut kept = Corrections::default();
    let first = short.descend_from(&mut kept, value, slope, Vector::zero(5), Vec::new())?;
    let carried =
        short.descend_from(&mut kept, value, slope, first.solution.clone(), Vec::new())?;
    let fresh = short.descend(value, slope, first.solution, Vec::new())?;
    assert!(carried.value < 0.5 * fresh.value);
    Ok(())
}

#[test]
fn the_wolfe_conditions_can_search() -> Result<(), AssertionError> {
    let solver = solver().with_line_search(Wolfe {
        control_1: 1.0e-4,
        control_2: 0.9,
        cut_back: 0.5,
        max_steps: 27,
        strong: false,
    });
    Assert::default().eq_within_tols(
        &solver
            .descend(
                rosenbrock,
                rosenbrock_gradient,
                [-1.2, 1.0].into(),
                Vec::new(),
            )
            .map_err(|error| AssertionError::from(format!("{error}")))?
            .solution,
        &[1.0, 1.0].into(),
    )
}
