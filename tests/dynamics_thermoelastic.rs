#![cfg(feature = "fem")]

//! The stability checks of explicit thermoelastic-thermal dynamics, on two tetrahedra.

use conspire::{
    constitutive::{
        multiphysics::Canonical, solid::thermoelastic::AlmansiHamel, thermal::conduction::Fourier,
    },
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Tetrahedron, thermal::NodalTemperatures},
        boundary::{BoundaryConditions, SolidThermal},
        multiphysics::solid_thermal::thermoelastic_thermal_conduction::dynamics::{
            Checked, Stability, ThermoelasticThermalConductionDynamics, Unchecked,
            time_scale::ThermoelasticThermalConductionTimeScaleElements,
        },
        solid::Lumped,
    },
    math::{
        Quantity, Tensor,
        integrate::{DormandPrinceFixedStep, IntegrationError},
    },
    units::{
        Density, PowerPerLengthTemperature, ReciprocalTemperature, SpecificHeat, Stress,
        Temperature, Time, VolumetricHeatCapacity,
    },
};

const COORDINATES: [[f64; 3]; 5] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];
const STRETCH_RATE: f64 = 0.01;
const REFERENCE_TEMPERATURE: f64 = 300.0;

type Coupled = Canonical<AlmansiHamel, Fourier>;
type Body =
    Block<Coupled, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>, Quantity<VolumetricHeatCapacity>>;

fn reference() -> NodalReferenceCoordinates<3> {
    COORDINATES.into()
}

fn model() -> Model<Body, 3> {
    let coupled: Coupled = (
        AlmansiHamel {
            bulk_modulus: Stress::pascals(13.0e9),
            shear_modulus: Stress::pascals(3.0e9),
            coefficient_of_thermal_expansion: ReciprocalTemperature::per_kelvin(1.0e-5),
            reference_temperature: Temperature::kelvin(REFERENCE_TEMPERATURE),
        },
        Fourier {
            thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(50.0),
        },
    )
        .into();
    (
        Block::<Coupled, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>::from((
            coupled,
            Density::kilograms_per_cubic_meter(7.8e3),
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
            &reference(),
        ))
        .with_heat_capacity(
            Density::kilograms_per_cubic_meter(7.8e3)
                * SpecificHeat::joules_per_kilogram_kelvin(450.0),
        ),
        reference(),
    )
        .into()
}

fn temperatures() -> NodalTemperatures {
    COORDINATES
        .map(|_| Temperature::kelvin(REFERENCE_TEMPERATURE))
        .into_iter()
        .collect()
}

fn stretching() -> NodalVelocities<3> {
    COORDINATES
        .map(|coordinate| coordinate.map(|component| STRETCH_RATE * component))
        .into()
}

/// The two fastest time scales of the body, of the wave and of the diffusion.
fn scales() -> (f64, f64) {
    let (wave, diffusion) = model()
        .fastest_time_scales(
            &reference(),
            &NodalCoordinates::from(COORDINATES),
            &temperatures(),
        )
        .unwrap();
    (wave.value(), diffusion.value())
}

/// The final coordinates and temperatures of a run over the times, with a stability policy.
fn run(
    stability: impl Stability<Model<Body, 3>, Lumped, Lumped>,
    times: &[f64],
) -> Result<(NodalCoordinates<3>, NodalTemperatures), IntegrationError> {
    let (_, coordinates, _, _, temperatures, _) = model().integrate(
        stability,
        Lumped,
        Lumped,
        &DormandPrinceFixedStep::default(),
        &times
            .iter()
            .map(|&time| Time::seconds(time))
            .collect::<Vec<_>>(),
        (NodalCoordinates::from(COORDINATES), stretching()),
        temperatures(),
        &(),
        &BoundaryConditions::<SolidThermal<3>>::new(),
    )?;
    Ok((
        coordinates[coordinates.len() - 1].clone(),
        temperatures[temperatures.len() - 1].clone(),
    ))
}

fn checked(safety: f64, interval: usize) -> Checked {
    Checked { safety, interval }
}

#[test]
fn a_stable_step_gives_the_same_run_checked_and_unchecked() {
    let (wave, diffusion) = scales();
    let step = 0.1 * wave.min(diffusion);
    let times = (0..=10).map(|k| k as f64 * step).collect::<Vec<_>>();
    let (coordinates, temperatures) = run(checked(0.9, 1), &times).unwrap();
    let (expected_coordinates, expected_temperatures) = run(Unchecked, &times).unwrap();
    assert!(coordinates == expected_coordinates);
    assert!(temperatures == expected_temperatures);
}

#[test]
fn an_unstable_step_is_refused_when_checked_and_taken_when_unchecked() {
    let (wave, diffusion) = scales();
    let step = 10.0 * wave.min(diffusion);
    let times = [0.0, step, 2.0 * step];
    assert!(matches!(
        run(checked(1.0, 1), &times),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
    assert!(run(Unchecked, &times).is_ok());
}

#[test]
fn a_checked_run_needs_an_interval() {
    let (wave, diffusion) = scales();
    let step = 0.1 * wave.min(diffusion);
    assert!(matches!(
        run(checked(0.9, 0), &[0.0, step, 2.0 * step]),
        Err(IntegrationError::Intermediate(message)) if message.contains("at least one")
    ));
}
