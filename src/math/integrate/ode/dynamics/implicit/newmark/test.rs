use super::Newmark;
use crate::{
    math::{
        Quantity, Scalar, Tensor, TensorVector,
        integrate::{FixedStep, ImplicitDynamics, IntegrationError, Times},
        optimize::{EqualityConstraint, NewtonRaphson},
    },
    units::{Acceleration, Length, RateSquared, Time, Velocity},
};

type Solution = (
    Times,
    TensorVector<Quantity<Length>>,
    TensorVector<Quantity<Velocity>>,
    TensorVector<Quantity<Acceleration>>,
);

const STIFFNESS: Scalar = 4.0;

fn oscillate(integrator: &Newmark, final_time: Scalar) -> Result<Solution, IntegrationError> {
    integrator.integrate(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        |_: Quantity<Time>,
         x: &Quantity<Length>,
         _: &Quantity<Velocity>,
         a: &Quantity<Acceleration>| {
            Ok(a + x * Quantity::<RateSquared>::new(STIFFNESS))
        },
        |_: Quantity<Time>,
         _: &Quantity<Length>,
         _: &Quantity<Velocity>,
         dt: Quantity<Time>,
         beta: Scalar,
         _: Scalar| {
            Ok(Quantity::<RateSquared>::new(
                1.0 / (beta * dt.value().powi(2)) + STIFFNESS,
            ))
        },
        NewtonRaphson::default(),
        vec![EqualityConstraint::None],
        None,
        &[Quantity::new(0.0), Quantity::new(final_time)],
        Quantity::new(1.0),
        Quantity::new(0.0),
    )
}

fn energy(x: &Quantity<Length>, v: &Quantity<Velocity>) -> Scalar {
    0.5 * (v.value().powi(2) + STIFFNESS * x.value().powi(2))
}

fn max_error(dt: Scalar) -> Scalar {
    let (time, x, _, _) = oscillate(&Newmark::new(Quantity::<Time>::new(dt)), 2.0).unwrap();
    time.iter()
        .zip(x.iter())
        .map(|(t, x)| (x.value() - (STIFFNESS.sqrt() * t.value()).cos()).abs())
        .fold(0.0, Scalar::max)
}

#[test]
fn fixed_step() {
    let integrator = Newmark::new(Quantity::<Time>::new(0.01));
    assert_eq!(FixedStep::<Time>::dt(&integrator).value(), 0.01);
}

#[test]
fn second_order() {
    let ratio = max_error(0.1) / max_error(0.05);
    assert!((ratio - 4.0).abs() < 0.2, "ratio {ratio}");
}

#[test]
fn conserves_energy_beyond_the_explicit_limit() {
    let dt = 2.0;
    let (_, x, v, _) = oscillate(&Newmark::new(Quantity::<Time>::new(dt)), 40.0).unwrap();
    let initial = energy(&x[0], &v[0]);
    x.iter().zip(v.iter()).for_each(|(x, v)| {
        assert!((energy(x, v) - initial).abs() < 1e-10 * initial);
    });
}

#[test]
fn damped_by_numerical_dissipation_when_gamma_exceeds_one_half() {
    let integrator = Newmark::with_parameters(Quantity::<Time>::new(0.1), 0.3025, 0.6).unwrap();
    let (_, x, v, _) = oscillate(&integrator, 10.0).unwrap();
    let last = x.len() - 1;
    assert!(energy(&x[last], &v[last]) < 0.9 * energy(&x[0], &v[0]));
}

#[test]
fn time_step_not_set() {
    assert!(matches!(
        oscillate(&Newmark::default(), 1.0),
        Err(IntegrationError::TimeStepNotSet(..))
    ));
}

#[test]
fn conditionally_stable_parameters_are_rejected() {
    let dt = Quantity::<Time>::new(0.1);
    [(0.0, 0.5), (0.2, 0.5), (0.25, 0.4), (0.25, 0.6)]
        .iter()
        .for_each(|&(beta, gamma)| {
            assert!(matches!(
                Newmark::with_parameters(dt, beta, gamma),
                Err(IntegrationError::InvalidNewmarkParameters(..))
            ));
        });
    assert!(Newmark::with_parameters(dt, 0.25, 0.5).is_ok());
    assert!(Newmark::with_parameters(dt, 0.3025, 0.6).is_ok());
}
