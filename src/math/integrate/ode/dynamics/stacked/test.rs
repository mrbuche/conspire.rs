use crate::{
    math::{
        Quantity, Scalar, Tensor, TensorVector,
        integrate::{
            ButcherTableau, DormandPrince, DormandPrinceFixedStep, DormandPrinceTableau,
            ExplicitDynamics, IntegrationError, Times, Verner9FixedStep,
        },
    },
    units::{Acceleration, Length, RateSquared, Time, Velocity},
};
use std::f64::consts::FRAC_PI_2;

type Solution = (
    Times,
    TensorVector<Quantity<Length>>,
    TensorVector<Quantity<Velocity>>,
    TensorVector<Quantity<Acceleration>>,
);

const STIFFNESS: Scalar = 4.0;

fn oscillate(
    integrator: &impl ExplicitDynamics<
        Quantity<Length>,
        TensorVector<Quantity<Length>>,
        TensorVector<Quantity<Velocity>>,
        TensorVector<Quantity<Acceleration>>,
    >,
    time: &[Quantity<Time>],
) -> Result<Solution, IntegrationError> {
    integrator.integrate(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        time,
        Quantity::new(1.0),
        Quantity::new(0.0),
    )
}

fn grid(steps: usize, final_time: Scalar) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Quantity::new(final_time * step as Scalar / steps as Scalar))
        .collect()
}

fn max_error(solution: &Solution) -> Scalar {
    let (time, x, ..) = solution;
    time.iter()
        .zip(x.iter())
        .map(|(t, x)| (x.value() - (STIFFNESS.sqrt() * t.value()).cos()).abs())
        .fold(0.0, Scalar::max)
}

#[test]
fn fixed_step_matches_exact_solution() {
    let solution = oscillate(&DormandPrinceFixedStep::default(), &grid(400, 2.0)).unwrap();
    assert!(max_error(&solution) < 1e-9);
}

#[test]
fn high_order_fixed_step_matches_exact_solution() {
    let solution = oscillate(&Verner9FixedStep::default(), &grid(100, 2.0)).unwrap();
    assert!(max_error(&solution) < 1e-9);
}

#[test]
fn variable_step_matches_exact_solution() {
    let (time, x, v, a) = oscillate(
        &DormandPrince::default(),
        &[Quantity::new(0.0), Quantity::new(2.0)],
    )
    .unwrap();
    assert!(time.len() > 2);
    assert_eq!(
        (x.len(), v.len(), a.len()),
        (time.len(), time.len(), time.len())
    );
    assert!(max_error(&(time, x, v, a)) < 1e-4);
}

#[test]
fn solution_is_consistent() {
    let (time, x, v, a) = oscillate(
        &DormandPrince::default(),
        &[Quantity::new(0.0), Quantity::new(1.0)],
    )
    .unwrap();
    assert_eq!(x[0].value(), 1.0);
    assert_eq!(v[0].value(), 0.0);
    assert_eq!(x.len(), time.len());
    x.iter()
        .zip(v.iter())
        .zip(a.iter())
        .for_each(|((x, v), a)| {
            assert!((a.value() + STIFFNESS * x.value()).abs() < 1e-9);
            assert!(v.value().abs() <= STIFFNESS.sqrt() + 1e-9);
        });
}

#[test]
fn bounded_unstable_step_errors() {
    let limit = 2.0 / STIFFNESS.sqrt();
    let result: Result<Solution, _> = DormandPrinceFixedStep::default().integrate_bounded(
        |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
        },
        |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
            Ok(Quantity::new(1.0 / STIFFNESS.sqrt()))
        },
        1.0,
        &grid(4, 4.0 * 4.0 * limit),
        Quantity::new(1.0),
        Quantity::new(0.0),
    );
    match result {
        Err(IntegrationError::UnstableTimeStep(_, reported, _)) => {
            let extent = DormandPrinceTableau::stability().extent(FRAC_PI_2);
            assert!((reported - extent / STIFFNESS.sqrt()).abs() < 1e-12);
        }
        _ => panic!("expected an unstable time step"),
    }
}

#[test]
fn bounded_stable_step_agrees() {
    let time = grid(400, 2.0);
    let bounded: Solution = DormandPrinceFixedStep::default()
        .integrate_bounded(
            |_: Quantity<Time>, x: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(x * Quantity::<RateSquared>::new(-STIFFNESS))
            },
            |_: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
                Ok(Quantity::new(1.0 / STIFFNESS.sqrt()))
            },
            0.9,
            &time,
            Quantity::new(1.0),
            Quantity::new(0.0),
        )
        .unwrap();
    let unbounded = oscillate(&DormandPrinceFixedStep::default(), &time).unwrap();
    assert_eq!(
        bounded.1[bounded.1.len() - 1].value(),
        unbounded.1[unbounded.1.len() - 1].value()
    );
}

#[test]
fn acceleration_error_is_reported() {
    let result: Result<Solution, _> = DormandPrinceFixedStep::default().integrate(
        |t: Quantity<Time>, _: &Quantity<Length>, _: &Quantity<Velocity>| {
            if t.value() > 0.25 {
                Err("failed".to_string())
            } else {
                Ok(Quantity::new(0.0))
            }
        },
        &grid(10, 1.0),
        Quantity::new(0.0),
        Quantity::new(0.0),
    );
    assert!(result.is_err());
}
