use super::VelocityVerlet;
use crate::{
    math::{
        Quantity, Scalar, Tensor, TensorVector,
        integrate::{ExplicitDynamics, FixedStep, IntegrationError, Times},
    },
    units::{Acceleration, Length, RateSquared, Time, Velocity},
};
use std::f64::consts::TAU;

type Solution = (
    Times,
    TensorVector<Quantity<Length>>,
    TensorVector<Quantity<Velocity>>,
    TensorVector<Quantity<Acceleration>>,
);

const STIFFNESS: Scalar = 4.0;

fn oscillate(
    integrator: &VelocityVerlet,
    final_time: Scalar,
) -> Result<Solution, IntegrationError> {
    integrator.integrate(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        &[Quantity::new(0.0), Quantity::new(final_time)],
        Quantity::new(1.0),
        Quantity::new(0.0),
    )
}

fn energy(x: &Quantity<Length>, v: &Quantity<Velocity>) -> Scalar {
    0.5 * (v.value().powi(2) + STIFFNESS * x.value().powi(2))
}

fn max_error(dt: Scalar) -> Scalar {
    let (time, x, _, _) = oscillate(&VelocityVerlet { dt }, 2.0).unwrap();
    time.iter()
        .zip(x.iter())
        .map(|(t, x)| (x.value() - (STIFFNESS.sqrt() * t.value()).cos()).abs())
        .fold(0.0, Scalar::max)
}

#[test]
fn fixed_step() {
    let integrator = VelocityVerlet { dt: 0.01 };
    assert_eq!(FixedStep::<Time>::dt(&integrator).value(), 0.01);
}

#[test]
fn oscillator_matches_exact_solution() {
    assert!(max_error(0.001) < 1e-5);
}

#[test]
fn second_order_convergence() {
    let ratio = max_error(0.01) / max_error(0.005);
    assert!((ratio - 4.0).abs() < 0.1, "ratio {ratio}");
}

#[test]
fn accelerations_are_consistent() {
    let (_, x, v, a) = oscillate(&VelocityVerlet { dt: 0.01 }, 1.0).unwrap();
    x.iter().zip(a.iter()).for_each(|(x, a)| {
        assert!((a.value() + STIFFNESS * x.value()).abs() < 1e-12);
    });
    assert_eq!(v[0].value(), 0.0);
}

#[test]
fn energy_stays_bounded_over_many_periods() {
    let periods = 1000.0;
    let (_, x, v, _) = oscillate(
        &VelocityVerlet { dt: 0.05 },
        periods * TAU / STIFFNESS.sqrt(),
    )
    .unwrap();
    let initial = energy(&x[0], &v[0]);
    let drift = x
        .iter()
        .zip(v.iter())
        .map(|(x, v)| (energy(x, v) - initial).abs() / initial)
        .fold(0.0, Scalar::max);
    assert!(drift < 1e-2, "drift {drift}");
}

#[test]
fn explicit_time_sequence() {
    let (time, x, v, a): Solution = VelocityVerlet::default()
        .integrate(
            |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
            },
            &[0.0, 0.1, 0.3, 0.4].map(Quantity::new),
            Quantity::new(1.0),
            Quantity::new(0.0),
        )
        .unwrap();
    assert_eq!((time.len(), x.len(), v.len(), a.len()), (4, 4, 4, 4));
    assert_eq!(time[2].value(), 0.3);
}

#[test]
fn bounded_stable_step_agrees() {
    let integrator = VelocityVerlet { dt: 0.01 };
    let bound = |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
        Ok(Quantity::new(1.0 / STIFFNESS.sqrt()))
    };
    let (_, x, ..): Solution = integrator
        .integrate_bounded(
            |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
            },
            bound,
            0.9,
            &[Quantity::new(0.0), Quantity::new(1.0)],
            Quantity::new(1.0),
            Quantity::new(0.0),
        )
        .unwrap();
    let (_, y, ..) = oscillate(&integrator, 1.0).unwrap();
    assert_eq!(x.len(), y.len());
    assert_eq!(x[x.len() - 1].value(), y[y.len() - 1].value());
}

#[test]
fn bounded_unstable_step_errors() {
    let limit = 2.0 / STIFFNESS.sqrt();
    let result: Result<Solution, _> = VelocityVerlet { dt: 1.1 * limit }.integrate_bounded(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(Quantity::new(1.0 / STIFFNESS.sqrt()))
        },
        1.0,
        &[Quantity::new(0.0), Quantity::new(10.0)],
        Quantity::new(1.0),
        Quantity::new(0.0),
    );
    assert!(matches!(
        result,
        Err(IntegrationError::UnstableTimeStep(..))
    ));
}

#[test]
fn bounded_step_at_limit_is_allowed_but_not_above_safety() {
    let limit = 2.0 / STIFFNESS.sqrt();
    let run = |dt: Scalar, safety: Scalar| -> Result<Solution, IntegrationError> {
        VelocityVerlet { dt }.integrate_bounded(
            |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
            },
            |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(Quantity::new(1.0 / STIFFNESS.sqrt()))
            },
            safety,
            &[Quantity::new(0.0), Quantity::new(1.0)],
            Quantity::new(1.0),
            Quantity::new(0.0),
        )
    };
    assert!(run(0.5 * limit, 0.5).is_ok());
    assert!(run(0.6 * limit, 0.5).is_err());
}

#[test]
fn invalid_safety_factor() {
    let result: Result<Solution, _> = VelocityVerlet { dt: 0.01 }.integrate_bounded(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| Ok(Quantity::new(1.0)),
        1.5,
        &[Quantity::new(0.0), Quantity::new(1.0)],
        Quantity::new(1.0),
        Quantity::new(0.0),
    );
    assert!(matches!(
        result,
        Err(IntegrationError::InvalidSafetyFactor(_))
    ));
}

#[test]
fn time_step_not_set() {
    assert!(matches!(
        oscillate(&VelocityVerlet::default(), 1.0),
        Err(IntegrationError::TimeStepNotSet(..))
    ));
}

#[test]
fn time_validation() {
    let integrate = |time: &[Scalar]| -> Result<Solution, IntegrationError> {
        VelocityVerlet { dt: 0.1 }.integrate(
            |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| panic!(),
            &time.iter().map(|t| Quantity::new(*t)).collect::<Vec<_>>(),
            Quantity::new(0.0),
            Quantity::new(0.0),
        )
    };
    assert!(matches!(
        integrate(&[0.0]),
        Err(IntegrationError::LengthTimeLessThanTwo)
    ));
    assert!(matches!(
        integrate(&[1.0, 0.0]),
        Err(IntegrationError::InitialTimeNotLessThanFinalTime)
    ));
}

#[test]
fn acceleration_error_is_reported() {
    let result: Result<Solution, _> = VelocityVerlet { dt: 0.1 }.integrate(
        |t: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
            if t.value() > 0.25 {
                Err("failed".to_string())
            } else {
                Ok(Quantity::new(0.0))
            }
        },
        &[Quantity::new(0.0), Quantity::new(1.0)],
        Quantity::new(0.0),
        Quantity::new(0.0),
    );
    assert!(matches!(result, Err(IntegrationError::Upstream(..))));
}
