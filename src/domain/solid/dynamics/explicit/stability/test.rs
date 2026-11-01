use super::{Checked, Stability, Unchecked};
use crate::{
    cbm::{
        ElasticExplicitDynamics, Execution, Lumped, Model, NodalCoordinates,
        NodalReferenceCoordinates, NodalVelocities, block::Block,
    },
    constitutive::solid::hyperelastic::NeoHookean,
    domain::solid::{dynamics::test::fixed_dofs, time_scale::TimeScaleElements},
    math::{
        Quantity,
        integrate::{IntegrationError, VelocityVerlet},
    },
    units::{Density, Stress, Time},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

const COORDINATES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

type B = Block<NeoHookean, Quantity<Density>>;

fn model() -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(COORDINATES);
    (
        (
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            DENSITY,
            vec![[0, 1, 2, 3], [1, 2, 3, 4]].into(),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

#[test]
fn a_bounded_integration_runs_below_the_limit_and_errors_above_it() {
    let model = model();
    let limit = model
        .fastest_time_scale(
            &NodalReferenceCoordinates::from(COORDINATES),
            &NodalCoordinates::from(COORDINATES),
        )
        .unwrap()
        .in_seconds();
    let run = |dt: f64| {
        model.integrate(
            Execution::Serial,
            Checked {
                safety: 1.0,
                interval: 1,
            },
            Lumped,
            &VelocityVerlet::new(Time::seconds(dt)),
            &[Time::seconds(0.0), Time::seconds(10.0 * dt)],
            (
                NodalCoordinates::from(COORDINATES),
                NodalVelocities::from(COORDINATES.map(|_| [0.0; 3])),
            ),
            &(),
            &fixed_dofs(&[]),
        )
    };
    assert!(run(0.5 * limit).is_ok());
    assert!(run(4.0 * limit).is_err());
}

fn rest(
    model: &Model<B, 3>,
    stability: impl Stability<Model<B, 3>, Lumped>,
    dt: f64,
) -> Result<(), IntegrationError> {
    model
        .integrate(
            Execution::Serial,
            stability,
            Lumped,
            &VelocityVerlet::new(Time::seconds(dt)),
            &[Time::seconds(0.0), Time::seconds(10.0 * dt)],
            (
                NodalCoordinates::from(COORDINATES),
                NodalVelocities::from(COORDINATES.map(|_| [0.0; 3])),
            ),
            &(),
            &fixed_dofs(&[]),
        )
        .map(|_| ())
}

fn limit() -> f64 {
    model()
        .fastest_time_scale(
            &NodalReferenceCoordinates::from(COORDINATES),
            &NodalCoordinates::from(COORDINATES),
        )
        .unwrap()
        .in_seconds()
}

#[test]
fn an_unchecked_integration_accepts_a_step_that_a_checked_one_refuses() {
    let (model, step) = (model(), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1,
    };
    assert!(matches!(
        rest(&model, checked, step),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
    assert!(rest(&model, Unchecked, step).is_ok());
}

#[test]
fn a_smaller_fraction_of_the_limit_refuses_a_step_that_a_larger_one_accepts() {
    let (model, step) = (model(), 1.5 * limit());
    let checked = |safety| Checked {
        safety,
        interval: 1,
    };
    assert!(rest(&model, checked(1.0), step).is_ok());
    assert!(rest(&model, checked(0.5), step).is_err());
}

#[test]
fn a_checked_integration_needs_an_interval() {
    let model = model();
    let checked = Checked {
        safety: 1.0,
        interval: 0,
    };
    assert!(matches!(
        rest(&model, checked, 0.5 * limit()),
        Err(IntegrationError::Intermediate(message)) if message.contains("at least one")
    ));
}

#[test]
fn a_checked_integration_with_a_longer_interval_still_checks_the_first_step() {
    let (model, step) = (model(), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1000,
    };
    assert!(rest(&model, checked, step).is_err());
}
