use crate::{
    constitutive::thermal::conduction::Fourier,
    domain::time_scale::largest_eigenvalue,
    fem::{
        Model, NodalReferenceCoordinates,
        block::{
            Block,
            element::linear::Tetrahedron,
            thermal::{NodalTemperatures, conduction::NodalForcesThermal},
        },
        thermal::{
            capacity::HeatCapacityMatrix, conduction::ThermalConductionElements,
            dynamics::ThermalConductionDynamics, time_scale::ThermalTimeScaleElements,
        },
    },
    math::{
        Hessian, Quantity, Tensor,
        integrate::{Euler, IntegrationError},
        optimize::EqualityConstraint,
    },
    units::{
        Density, PowerPerLengthTemperature, SpecificHeat, Temperature, Time, VolumetricHeatCapacity,
    },
};

const NODES: usize = 5;
const CONNECTIVITY: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 3, 4]];

fn coordinates() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from([
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ])
}

type B = Block<Fourier, Tetrahedron<4>, 4, 3, 4, 4, Quantity<VolumetricHeatCapacity>>;

fn model() -> Model<B, 3> {
    (
        Block::<Fourier, Tetrahedron<4>, 4, 3, 4, 4>::from((
            Fourier {
                thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(50.0),
            },
            CONNECTIVITY.to_vec(),
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

fn temperatures(kelvin: [f64; NODES]) -> NodalTemperatures {
    kelvin.map(Temperature::kelvin).into()
}

fn times(dt: f64, steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Time::seconds(step as f64 * dt))
        .collect()
}

fn values(temperatures: &NodalTemperatures) -> Vec<f64> {
    temperatures.iter().map(|t| t.value()).collect()
}

fn near(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{a} is not within {tolerance} of {b}"
    );
}

mod time_scale {
    use super::*;
    #[test]
    fn bounds_the_global_time_scale_from_above_and_is_not_loose() {
        let model = model();
        let initial = temperatures([300.0; NODES]);
        let stiffnesses = model.nodal_stiffnesses(&initial).unwrap();
        let capacities: Vec<f64> = model
            .nodal_lumped_heat_capacities()
            .iter()
            .map(|capacity| capacity.value())
            .collect();
        let global = 1.0
            / largest_eigenvalue(
                NODES,
                |row, column| stiffnesses.entry(row, column),
                &capacities,
            );
        let elements = model
            .fastest_diffusive_time_scale(&initial)
            .unwrap()
            .value();
        assert!(elements <= global * (1.0 + 1e-6), "{elements} > {global}");
        assert!(elements >= 0.2 * global, "{elements} << {global}");
    }
}

mod integrate_temperatures {
    use super::*;
    const SAFETY: f64 = 0.5;
    fn step(model: &Model<B, 3>) -> f64 {
        0.4 * model
            .fastest_diffusive_time_scale(&temperatures([300.0; NODES]))
            .unwrap()
            .value()
    }
    #[test]
    fn an_insulated_body_conserves_its_heat_and_relaxes_to_the_mean() {
        let model = model();
        let dt = step(&model);
        let capacities = model.nodal_lumped_heat_capacities();
        let heat = |temperatures: &NodalTemperatures| -> f64 {
            capacities
                .iter()
                .zip(temperatures.iter())
                .map(|(capacity, temperature)| capacity.value() * temperature.value())
                .sum()
        };
        let initial = temperatures([300.0, 500.0, 350.0, 420.0, 280.0]);
        let (.., history, _) = {
            let (times, history, rates) = model
                .integrate_bounded(
                    &Euler::default(),
                    SAFETY,
                    1,
                    &times(dt, 4000),
                    initial.clone(),
                    &NodalForcesThermal::zero(NODES),
                    &capacities,
                    EqualityConstraint::None,
                )
                .unwrap();
            (times, history, rates)
        };
        let last = history.iter().last().unwrap();
        near(heat(last), heat(&initial), 1e-9);
        let mean = heat(&initial) / capacities.iter().map(|c| c.value()).sum::<f64>();
        values(last)
            .iter()
            .for_each(|&temperature| near(temperature, mean, 1e-6));
    }
    #[test]
    fn a_fixed_temperature_is_held_and_the_rest_relax_to_it() {
        let model = model();
        let dt = step(&model);
        let (_, history, rates) = model
            .integrate_bounded(
                &Euler::default(),
                SAFETY,
                1,
                &times(dt, 4000),
                temperatures([300.0, 500.0, 350.0, 420.0, 280.0]),
                &NodalForcesThermal::zero(NODES),
                &model.nodal_lumped_heat_capacities(),
                EqualityConstraint::Fixed(vec![0]),
            )
            .unwrap();
        history.iter().for_each(|t| assert_eq!(t[0].value(), 300.0));
        rates.iter().for_each(|r| assert_eq!(r[0].value(), 0.0));
        values(history.iter().last().unwrap())
            .iter()
            .for_each(|&temperature| near(temperature, 300.0, 1e-6));
    }
    #[test]
    fn consistent_and_lumped_capacities_reach_the_same_steady_state() {
        let model = model();
        let dt = step(&model) / 4.0;
        let heating = NodalForcesThermal::from(
            [0.0, 0.0, 2.0e3, 0.0, 1.0e3]
                .map(|watts| crate::units::Energy::joules(watts) / Time::seconds(1.0)),
        );
        let fixed = || EqualityConstraint::Fixed(vec![0]);
        let initial = || temperatures([300.0; NODES]);
        let (_, lumped, _) = model
            .integrate(
                &Euler::default(),
                &times(dt, 20000),
                initial(),
                &heating,
                &model.nodal_lumped_heat_capacities(),
                fixed(),
            )
            .unwrap();
        let (_, consistent, _) = model
            .integrate(
                &Euler::default(),
                &times(dt, 20000),
                initial(),
                &heating,
                &model.nodal_heat_capacities(),
                fixed(),
            )
            .unwrap();
        let (lumped, consistent) = (
            values(lumped.iter().last().unwrap()),
            values(consistent.iter().last().unwrap()),
        );
        lumped
            .iter()
            .zip(&consistent)
            .for_each(|(&a, &b)| near(a, b, 1e-6));
        assert!(lumped[2] > 300.0 && lumped[4] > 300.0);
        let rates = model
            .nodal_temperature_rates(
                &temperatures(lumped.clone().try_into().unwrap()),
                &heating,
                &model.nodal_lumped_heat_capacities().inverse(&[0]).unwrap(),
            )
            .unwrap();
        rates.iter().for_each(|rate| near(rate.value(), 0.0, 1e-6));
    }
    #[test]
    fn a_step_above_the_limit_is_an_error_and_would_diverge() {
        let model = model();
        let dt = 3.0
            * model
                .fastest_diffusive_time_scale(&temperatures([300.0; NODES]))
                .unwrap()
                .value();
        let initial = temperatures([300.0, 500.0, 350.0, 420.0, 280.0]);
        let capacities = model.nodal_lumped_heat_capacities();
        let error = model
            .integrate_bounded(
                &Euler::default(),
                1.0,
                1,
                &times(dt, 40),
                initial.clone(),
                &NodalForcesThermal::zero(NODES),
                &capacities,
                EqualityConstraint::None,
            )
            .unwrap_err();
        assert!(matches!(error, IntegrationError::UnstableTimeStep(..)));
        let (_, history, _) = model
            .integrate(
                &Euler::default(),
                &times(20.0 * dt, 40),
                initial,
                &NodalForcesThermal::zero(NODES),
                &capacities,
                EqualityConstraint::None,
            )
            .unwrap();
        assert!(
            values(history.iter().last().unwrap())
                .iter()
                .any(|temperature| temperature.abs() > 1e6)
        );
    }
}
