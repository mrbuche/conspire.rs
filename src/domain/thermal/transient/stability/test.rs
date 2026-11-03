use super::{Checked, Stability, Unchecked};
use crate::{
    cbm::{
        Lumped, Model,
        block::Block,
        thermal::{time_scale::ThermalTimeScaleElements, transient::ThermalConductionTransient},
    },
    constitutive::thermal::conduction::Fourier,
    domain::thermal::{NodalTemperatures, transient::test::fixed_temperatures},
    math::{
        Quantity,
        integrate::{Euler, IntegrationError},
    },
    units::{
        Density, PowerPerLengthTemperature, SpecificHeat, Temperature, Time, VolumetricHeatCapacity,
    },
};

const NODES: usize = 5;

const COORDINATES: [[f64; 3]; NODES] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

type B = Block<Fourier, Quantity<VolumetricHeatCapacity>>;

fn model() -> Model<B, 3> {
    let reference = COORDINATES.into();
    (
        Block::<Fourier>::from((
            Fourier {
                thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(50.0),
            },
            vec![[0, 1, 2, 3], [1, 2, 3, 4]].into(),
            &reference,
        ))
        .with_heat_capacity(
            Density::kilograms_per_cubic_meter(7.8e3)
                * SpecificHeat::joules_per_kilogram_kelvin(450.0),
        ),
        reference,
    )
        .into()
}

fn temperatures() -> NodalTemperatures {
    [300.0, 500.0, 350.0, 420.0, 280.0]
        .map(Temperature::kelvin)
        .into()
}

fn limit() -> f64 {
    let model = model();
    model
        .fastest_diffusive_time_scale(&model.coordinates, &temperatures())
        .unwrap()
        .value()
}

fn run(
    model: &Model<B, 3>,
    stability: impl Stability<Model<B, 3>, Lumped>,
    dt: f64,
) -> Result<(), IntegrationError> {
    model
        .integrate(
            stability,
            Lumped,
            &Euler::default(),
            &(0..=10)
                .map(|step| Time::seconds(step as f64 * dt))
                .collect::<Vec<_>>(),
            temperatures(),
            &(),
            &fixed_temperatures(&[]),
        )
        .map(|_| ())
}

#[test]
fn a_checked_run_refuses_a_step_above_the_limit_and_accepts_one_below_it() {
    let model = model();
    let checked = Checked {
        safety: 1.0,
        interval: 1,
    };
    assert!(run(&model, checked, 0.4 * limit()).is_ok());
    assert!(matches!(
        run(&model, checked, 4.0 * limit()),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
}

#[test]
fn an_unchecked_run_accepts_a_step_that_a_checked_one_refuses() {
    let (model, step) = (model(), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1,
    };
    assert!(run(&model, checked, step).is_err());
    assert!(run(&model, Unchecked, step).is_ok());
}

#[test]
fn a_smaller_fraction_of_the_limit_refuses_a_step_that_a_larger_one_accepts() {
    let (model, step) = (model(), 1.5 * limit());
    let checked = |safety| Checked {
        safety,
        interval: 1,
    };
    assert!(run(&model, checked(1.0), step).is_ok());
    assert!(run(&model, checked(0.5), step).is_err());
}

#[test]
fn a_checked_run_needs_an_interval() {
    let model = model();
    let checked = Checked {
        safety: 1.0,
        interval: 0,
    };
    assert!(matches!(
        run(&model, checked, 0.4 * limit()),
        Err(IntegrationError::Intermediate(message)) if message.contains("at least one")
    ));
}

#[test]
fn a_checked_run_with_a_longer_interval_still_checks_the_first_step() {
    let (model, step) = (model(), 4.0 * limit());
    let checked = Checked {
        safety: 1.0,
        interval: 1000,
    };
    assert!(run(&model, checked, step).is_err());
}
