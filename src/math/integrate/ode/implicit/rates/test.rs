use crate::{
    math::{
        Quantity, Scalar, Tensor, TensorVector,
        integrate::{BackwardEuler, ImplicitRates, IntegrationError, Times},
        optimize::{EqualityConstraint, NewtonRaphson},
    },
    units::{Length, Rate, Time, Velocity},
};

type Solution = (
    Times,
    TensorVector<Quantity<Length>>,
    TensorVector<Quantity<Velocity>>,
);

const CAPACITY: Scalar = 2.0;
const STIFFNESS: Scalar = 3.0;

fn decay(integrator: &BackwardEuler, final_time: Scalar) -> Result<Solution, IntegrationError> {
    integrator.integrate(
        |_: Quantity<Time>, x: &Quantity<Length>| {
            Ok(x * Quantity::<Rate>::new(-STIFFNESS / CAPACITY))
        },
        |_: Quantity<Time>, x: &Quantity<Length>, r: &Quantity<Velocity>| {
            Ok(r * CAPACITY + x * Quantity::<Rate>::new(STIFFNESS))
        },
        |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>, dt: Quantity<Time>| {
            Ok(Quantity::<Rate>::new(CAPACITY / dt.value() + STIFFNESS))
        },
        NewtonRaphson::default(),
        |_| vec![EqualityConstraint::None],
        None,
        &[Quantity::new(0.0), Quantity::new(final_time)],
        Quantity::new(1.0),
    )
}

fn max_error(dt: Scalar) -> Scalar {
    let (time, x, _) = decay(&BackwardEuler { dt }, 1.0).unwrap();
    time.iter()
        .zip(x.iter())
        .map(|(t, x)| (x.value() - (-STIFFNESS / CAPACITY * t.value()).exp()).abs())
        .fold(0.0, Scalar::max)
}

#[test]
fn first_order() {
    let ratio = max_error(0.02) / max_error(0.01);
    assert!((ratio - 2.0).abs() < 0.1, "ratio {ratio}");
}

#[test]
fn stable_beyond_the_explicit_limit() {
    let dt = 10.0 * CAPACITY / STIFFNESS;
    let (_, x, _) = decay(&BackwardEuler { dt }, 5.0 * dt).unwrap();
    x.iter()
        .zip(x.iter().skip(1))
        .for_each(|(a, b)| assert!(b.value() > 0.0 && b.value() < a.value()));
}

#[test]
fn rates_satisfy_the_system() {
    let (_, x, r) = decay(&BackwardEuler { dt: 0.1 }, 1.0).unwrap();
    x.iter().zip(r.iter()).skip(1).for_each(|(x, r)| {
        assert!((r.value() * CAPACITY + x.value() * STIFFNESS).abs() < 1e-8);
    });
}

#[test]
fn time_step_not_set() {
    assert!(matches!(
        decay(&BackwardEuler::default(), 1.0),
        Err(IntegrationError::TimeStepNotSet(..))
    ));
}
