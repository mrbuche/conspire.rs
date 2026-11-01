#![cfg(feature = "fem")]

//! The heat capacities of the temperatures of a body that conducts heat, on two tetrahedra.
//!
//! With no loads and no fixed temperatures, the heat in the body is conserved and its
//! temperatures relax to the one temperature that holds it, the mean of the initial ones
//! weighted by the lumped heat capacities, whichever heat capacities are used.

use conspire::{
    constitutive::thermal::conduction::Fourier,
    fem::{
        Model, NodalReferenceCoordinates,
        block::{Block, NoDensity, element::linear::Tetrahedron, thermal::NodalTemperatures},
        boundary::{BoundaryConditions, Field},
        solid::{Consistent, Lumped},
        thermal::{
            capacity::CapacityChoice,
            transient::{ThermalConductionImplicitTransient, ThermalConductionTransient},
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{BackwardEuler, Euler},
        optimize::{NewtonRaphson, Tolerances},
    },
    units::{
        Density, Dimensionless, PowerPerLengthTemperature, SpecificHeat, Temperature, Time,
        VolumetricHeatCapacity,
    },
};

const INITIAL: [f64; 5] = [300.0, 500.0, 350.0, 420.0, 280.0];

type Body = Block<Fourier, Tetrahedron<4>, 4, 3, 4, 4, NoDensity, Quantity<VolumetricHeatCapacity>>;

fn coordinates() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ])
}

fn model() -> Model<Body, 3> {
    (
        Block::<Fourier, Tetrahedron<4>, 4, 3, 4, 4>::from((
            Fourier {
                thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(50.0),
            },
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
            &coordinates(),
        ))
        .with_heat_capacity(
            Density::kilograms_per_cubic_meter(7.8e3)
                * SpecificHeat::joules_per_kilogram_kelvin(450.0),
        ),
        coordinates(),
    )
        .into()
}

fn initial() -> NodalTemperatures {
    INITIAL.map(Temperature::kelvin).into()
}

fn times(dt: f64, steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Time::seconds(step as f64 * dt))
        .collect()
}

/// The temperature that holds the heat of the initial ones, by the lumped heat capacities.
fn mean(model: &Model<Body, 3>) -> f64 {
    let capacities = model.nodal_lumped_heat_capacities();
    let heat = (0..INITIAL.len())
        .map(|node| capacities[node].value() * INITIAL[node])
        .sum::<f64>();
    heat / capacities
        .iter()
        .map(|capacity| capacity.value())
        .sum::<f64>()
}

fn solver() -> NewtonRaphson {
    NewtonRaphson {
        abs_tol: Tolerances {
            constraint: 1e-12,
            residual: 1e-6,
        },
        rel_tol: Some(Dimensionless::of(1e-12)),
        ..Default::default()
    }
}

/// The temperatures after steps of implicit integration.
fn implicitly(capacity: impl CapacityChoice<Model<Body, 3>>, dt: f64, steps: usize) -> Vec<f64> {
    let model = model();
    let (_, temperatures, _) = ThermalConductionImplicitTransient::integrate(
        &model,
        capacity,
        &BackwardEuler { dt },
        solver(),
        &times(dt, steps),
        initial(),
        &(),
        &BoundaryConditions::<Field<Temperature>>::new(),
    )
    .unwrap();
    temperatures
        .iter()
        .last()
        .unwrap()
        .iter()
        .map(|temperature| temperature.value())
        .collect()
}

/// The temperatures after a long time.
fn relaxed_implicitly(capacity: impl CapacityChoice<Model<Body, 3>>) -> Vec<f64> {
    implicitly(capacity, 5.0e3, 400)
}

/// The temperatures after a time, relaxing under explicit integration.
fn relaxed_explicitly(capacity: impl CapacityChoice<Model<Body, 3>>) -> Vec<f64> {
    let model = model();
    let (_, temperatures, _) = ThermalConductionTransient::integrate(
        &model,
        capacity,
        &Euler::default(),
        &times(1e-3, 4000),
        initial(),
        &(),
        &BoundaryConditions::<Field<Temperature>>::new(),
    )
    .unwrap();
    temperatures
        .iter()
        .last()
        .unwrap()
        .iter()
        .map(|temperature| temperature.value())
        .collect()
}

fn assert_relaxed_to_the_mean(temperatures: &[f64]) {
    let expected = mean(&model());
    temperatures.iter().enumerate().for_each(|(node, &found)| {
        assert!(
            (found - expected).abs() < 1e-6 * expected,
            "node {node}: {found} against {expected}"
        )
    });
}

#[test]
fn lumped_heat_capacities_relax_the_body_to_the_mean_temperature() {
    assert_relaxed_to_the_mean(&relaxed_implicitly(Lumped));
}

#[test]
fn consistent_heat_capacities_relax_the_body_to_the_mean_temperature() {
    assert_relaxed_to_the_mean(&relaxed_implicitly(Consistent));
}

#[test]
fn the_explicit_integration_conserves_the_heat_with_either_heat_capacities() {
    [relaxed_explicitly(Lumped), relaxed_explicitly(Consistent)]
        .iter()
        .for_each(|temperatures| {
            let capacities = model().nodal_lumped_heat_capacities();
            let heat = |temperatures: &[f64]| {
                (0..temperatures.len())
                    .map(|node| capacities[node].value() * temperatures[node])
                    .sum::<f64>()
            };
            let (found, expected) = (heat(temperatures), heat(&INITIAL));
            assert!(
                (found - expected).abs() < 1e-9 * expected,
                "{found} against {expected}"
            );
        });
}

#[test]
fn lumped_and_consistent_heat_capacities_differ_before_the_body_has_relaxed() {
    let (lumped, consistent) = (
        implicitly(Lumped, 1.0e3, 3),
        implicitly(Consistent, 1.0e3, 3),
    );
    let difference = lumped
        .iter()
        .zip(&consistent)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(difference > 1e-3, "the largest difference is {difference}");
}
