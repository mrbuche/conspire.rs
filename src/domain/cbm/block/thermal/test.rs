use super::NodalTemperatures;
use crate::{
    cbm::{
        Model, NodalReferenceCoordinates,
        block::{Block, node::thermal::ThermalNode},
        thermal::{
            conduction::ThermalConductionElements, dynamics::ThermalConductionDynamics,
            time_scale::ThermalTimeScaleElements,
        },
    },
    constitutive::thermal::conduction::Fourier,
    domain::{thermal::NodalForcesThermal, time_scale::largest_eigenvalue},
    geometry::mesh::PrimitiveConnectivity,
    math::{Hessian, Quantity, Tensor, integrate::Euler, optimize::EqualityConstraint},
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

fn conductivity() -> Fourier {
    Fourier {
        thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(50.0),
    }
}

fn heat_capacity() -> Quantity<VolumetricHeatCapacity> {
    Density::kilograms_per_cubic_meter(7.8e3) * SpecificHeat::joules_per_kilogram_kelvin(450.0)
}

fn reference() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from(COORDINATES)
}

fn block() -> B {
    Block::<Fourier>::from((
        conductivity(),
        PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 2, 3, 4]]),
        &reference(),
    ))
    .with_heat_capacity(heat_capacity())
}

fn model() -> Model<B, 3> {
    (block(), reference()).into()
}

fn temperatures(kelvin: [f64; NODES]) -> NodalTemperatures {
    kelvin.map(Temperature::kelvin).into()
}

fn times(dt: f64, steps: usize) -> Vec<Quantity<Time>> {
    (0..=steps)
        .map(|step| Time::seconds(step as f64 * dt))
        .collect()
}

fn near(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{a} is not within {tolerance} of {b}"
    );
}

mod conduction {
    use super::*;

    #[test]
    fn a_linear_temperature_field_has_its_gradient_at_every_particle() {
        let gradient = [3.0, -2.0, 5.0];
        let field = temperatures(
            COORDINATES
                .map(|[x, y, z]| 300.0 + gradient[0] * x + gradient[1] * y + gradient[2] * z),
        );
        block().nodes.iter().for_each(|node| {
            let found = node.temperature_gradient(&field);
            (0..3).for_each(|i| near(found[i].value(), gradient[i], 1e-10))
        });
    }

    #[test]
    fn the_nodal_heating_is_the_potential_gradient() {
        let model = model();
        let initial = temperatures([300.0, 500.0, 350.0, 420.0, 280.0]);
        let forces = model.nodal_forces(&initial).unwrap();
        let step = 1e-4;
        (0..NODES).for_each(|node| {
            let shifted = |sign: f64| {
                let mut shifted = initial.clone();
                shifted[node] += Temperature::kelvin(sign * step);
                model.potential(&shifted).unwrap().value()
            };
            near(
                (shifted(1.0) - shifted(-1.0)) / (2.0 * step),
                forces[node].value(),
                1e-6,
            )
        });
    }

    #[test]
    fn the_stiffness_is_the_symmetric_derivative_of_the_heating() {
        let model = model();
        let initial = temperatures([300.0, 500.0, 350.0, 420.0, 280.0]);
        let stiffnesses = model.nodal_stiffnesses(&initial).unwrap();
        let step = 1.0;
        (0..NODES).for_each(|column| {
            let shifted = |sign: f64| {
                let mut shifted = initial.clone();
                shifted[column] += Temperature::kelvin(sign * step);
                model.nodal_forces(&shifted).unwrap()
            };
            let (up, down) = (shifted(1.0), shifted(-1.0));
            (0..NODES).for_each(|row| {
                near(
                    (up[row].value() - down[row].value()) / (2.0 * step),
                    stiffnesses.entry(row, column),
                    1e-8,
                );
                near(
                    stiffnesses.entry(row, column),
                    stiffnesses.entry(column, row),
                    1e-12,
                )
            })
        });
    }

    #[test]
    fn heat_is_conserved_and_a_uniform_temperature_does_not_flow() {
        let model = model();
        let flows = model
            .nodal_forces(&temperatures([300.0, 500.0, 350.0, 420.0, 280.0]))
            .unwrap();
        near(
            flows.iter().map(|flow| flow.value()).sum::<f64>(),
            0.0,
            1e-9,
        );
        model
            .nodal_forces(&temperatures([300.0; NODES]))
            .unwrap()
            .iter()
            .for_each(|flow| near(flow.value(), 0.0, 1e-9));
    }
}

mod capacity {
    use super::*;

    #[test]
    fn the_nodal_capacities_total_the_volumetric_capacity_times_the_volume() {
        let model = model();
        let total: f64 = model
            .nodal_lumped_heat_capacities()
            .iter()
            .map(|capacity| capacity.value())
            .sum();
        let volume = block()
            .with_density(Density::kilograms_per_cubic_meter(1.0))
            .mass()
            .value();
        near(total, heat_capacity().value() * volume, 1e-10);
    }
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
        let patches = model
            .fastest_diffusive_time_scale(&initial)
            .unwrap()
            .value();
        assert!(patches <= global * (1.0 + 1e-6), "{patches} > {global}");
        assert!(patches >= 0.2 * global, "{patches} << {global}");
    }
}

mod integrate {
    use super::*;

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
        let (_, history, _) = model
            .integrate_bounded(
                &Euler::default(),
                0.5,
                1,
                &times(dt, 4000),
                initial.clone(),
                &NodalForcesThermal::zero(NODES),
                &capacities,
                EqualityConstraint::None,
            )
            .unwrap();
        let last = history.iter().last().unwrap();
        near(heat(last), heat(&initial), 1e-9);
        let mean = heat(&initial) / capacities.iter().map(|c| c.value()).sum::<f64>();
        last.iter()
            .for_each(|temperature| near(temperature.value(), mean, 1e-6));
    }

    #[test]
    fn a_fixed_temperature_is_held_and_the_rest_relax_to_it() {
        let model = model();
        let dt = step(&model);
        let (_, history, rates) = model
            .integrate_bounded(
                &Euler::default(),
                0.5,
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
        history
            .iter()
            .last()
            .unwrap()
            .iter()
            .for_each(|temperature| near(temperature.value(), 300.0, 1e-6));
    }
}
