#![cfg(feature = "fem")]

//! The masses and the stability checks of explicit dynamics, on two tetrahedra, the elements
//! that have consistent masses.

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Tetrahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            Checked, Consistent, ElasticExplicitDynamics, Execution, Lumped, Unchecked,
            time_scale::TimeScaleElements,
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, VelocityVerlet},
    },
    units::{Density, Stress, Time},
};

type Tetrahedra = Block<NeoHookean, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;

const GRAVITY: f64 = 9.81;
const STEPS: usize = 20;
const STEP: f64 = 1e-6;

fn coordinates() -> Vec<[f64; 3]> {
    vec![
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ]
}

fn model() -> Model<Tetrahedra, 3> {
    let reference = NodalReferenceCoordinates::from(coordinates());
    (
        Tetrahedra::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            Density::kilograms_per_cubic_meter(7.8e3),
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
            &reference,
        )),
        reference,
    )
        .into()
}

fn gravity() -> NodalAccelerations<3> {
    NodalAccelerations::from(vec![[0.0, 0.0, -GRAVITY]; coordinates().len()])
}

fn time() -> [Quantity<Time>; 2] {
    [Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)]
}

fn at_rest() -> (NodalCoordinates<3>, NodalVelocities<3>) {
    (
        NodalCoordinates::from(coordinates()),
        NodalVelocities::from(vec![[0.0; 3]; coordinates().len()]),
    )
}

/// The weight of the body, made from the masses of the choice, and its fall.
macro_rules! fall {
    ($model:expr, $execution:expr, $mass:expr, $masses:expr) => {{
        let weights = conspire::fem::boundary::Loads::<Solid<3>>::new()
            .body_acceleration(
                &NodalReferenceCoordinates::from(coordinates()),
                &$masses,
                &gravity()[0],
            )
            .unwrap();
        $model
            .integrate(
                $execution,
                Unchecked,
                $mass,
                &VelocityVerlet::new(Time::seconds(STEP)),
                &time(),
                at_rest(),
                &weights,
                &BoundaryConditions::<Solid<3>>::new(),
            )
            .unwrap()
    }};
}

fn assert_falls_with_gravity(
    (times, coordinates_history, _, accelerations): &(
        conspire::math::integrate::Times,
        conspire::fem::NodalCoordinatesHistory<3>,
        conspire::fem::NodalVelocitiesHistory<3>,
        conspire::fem::NodalAccelerationsHistory<3>,
    ),
) {
    let last = times.len() - 1;
    let elapsed = times[last].value();
    let reference = coordinates();
    (0..reference.len()).for_each(|node| {
        let fallen = reference[node][2] - 0.5 * GRAVITY * elapsed.powi(2);
        let z = coordinates_history[last][node][2].value();
        assert!(
            (z - fallen).abs() < 1e-12,
            "node {node}: {z} against {fallen}"
        );
        (0..3).for_each(|component| {
            let expected = if component == 2 { -GRAVITY } else { 0.0 };
            let found = accelerations[0][node][component].value();
            assert!(
                (found - expected).abs() < 1e-6,
                "acceleration of node {node}, component {component}: {found} against {expected}"
            );
        });
    });
}

#[test]
fn a_body_with_lumped_masses_falls_with_gravity() {
    let model = model();
    let solution = fall!(
        model,
        Execution::Serial,
        Lumped,
        model.nodal_lumped_masses()
    );
    assert_falls_with_gravity(&solution);
}

#[test]
fn a_body_with_consistent_masses_falls_with_gravity() {
    let model = model();
    let solution = fall!(model, Execution::Serial, Consistent, model.nodal_masses());
    assert_falls_with_gravity(&solution);
}

#[test]
fn lumped_and_consistent_masses_give_the_same_fall() {
    let model = model();
    let lumped = fall!(
        model,
        Execution::Serial,
        Lumped,
        model.nodal_lumped_masses()
    );
    let consistent = fall!(model, Execution::Serial, Consistent, model.nodal_masses());
    let last = lumped.0.len() - 1;
    (0..coordinates().len()).for_each(|node| {
        (0..3).for_each(|component| {
            let a = lumped.1[last][node][component].value();
            let b = consistent.1[last][node][component].value();
            assert!((a - b).abs() < 1e-12, "node {node}, component {component}");
        })
    });
}

#[test]
fn consistent_masses_give_the_same_on_threads() {
    let model = model();
    let serial = fall!(model, Execution::Serial, Consistent, model.nodal_masses());
    let threads = fall!(
        model,
        Execution::Threads(2),
        Consistent,
        model.nodal_masses()
    );
    let last = serial.0.len() - 1;
    assert!(serial.1[last] == threads.1[last]);
    assert!(serial.3[last] == threads.3[last]);
}

fn rest(
    model: &Model<Tetrahedra, 3>,
    stability: Checked,
    step: f64,
) -> Result<(), IntegrationError> {
    model
        .integrate(
            Execution::Serial,
            stability,
            Lumped,
            &VelocityVerlet::new(Time::seconds(step)),
            &[Time::seconds(0.0), Time::seconds(10.0 * step)],
            at_rest(),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .map(|_| ())
}

#[test]
fn a_checked_run_of_a_body_refuses_a_step_above_the_limit() {
    let model = model();
    let limit = model
        .fastest_time_scale(
            &NodalReferenceCoordinates::from(coordinates()),
            &NodalCoordinates::from(coordinates()),
        )
        .unwrap()
        .in_seconds();
    let checked = Checked {
        safety: 1.0,
        interval: 1,
    };
    assert!(rest(&model, checked, 0.5 * limit).is_ok());
    assert!(matches!(
        rest(&model, checked, 4.0 * limit),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
}
