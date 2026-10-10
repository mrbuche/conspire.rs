use crate::{
    constitutive::thermal::conduction::Fourier,
    math::{Quantity, Scalar},
    units::{PowerPerLengthTemperature, Temperature},
    vem::{
        NodalReferenceCoordinates,
        block::{
            Block,
            element::{DEFAULT_STABILIZATION, Element},
            thermal::NodalTemperatures,
        },
    },
};

pub(super) const CELLS: usize = 2;
pub(super) const NODES: usize = (CELLS + 1) * (CELLS + 1) * (CELLS + 1);
pub(super) const INTERIOR: usize = 13;
pub(super) const MODEL: Fourier = Fourier {
    thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(2.5),
};

pub(super) type B = Block<Fourier, Element>;

pub(super) fn node(i: usize, j: usize, k: usize) -> usize {
    i + (CELLS + 1) * (j + (CELLS + 1) * k)
}

pub(super) fn coordinates() -> Vec<[Scalar; 3]> {
    (0..NODES)
        .map(|index| {
            let (i, j, k) = (
                index % (CELLS + 1),
                (index / (CELLS + 1)) % (CELLS + 1),
                index / ((CELLS + 1) * (CELLS + 1)),
            );
            let bump = |axis: usize| ((index * 3 + axis) as Scalar * 5.3).sin() * 0.12;
            [
                i as Scalar + bump(0),
                j as Scalar + bump(1),
                k as Scalar + bump(2),
            ]
        })
        .collect()
}

pub(super) fn block() -> B {
    let corners = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    let local_faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];
    let mut elements_faces = Vec::new();
    let mut faces_nodes = Vec::new();
    (0..CELLS).for_each(|k| {
        (0..CELLS).for_each(|j| {
            (0..CELLS).for_each(|i| {
                let global = corners.map(|(a, b, c)| node(i + a, j + b, k + c));
                elements_faces.push(
                    local_faces
                        .iter()
                        .map(|face| {
                            faces_nodes.push(face.map(|corner| global[corner]).to_vec());
                            faces_nodes.len() - 1
                        })
                        .collect(),
                )
            })
        })
    });
    (
        MODEL,
        elements_faces,
        faces_nodes,
        &NodalReferenceCoordinates::from(coordinates()),
        DEFAULT_STABILIZATION,
    )
        .into()
}

pub(super) fn temperatures(values: &[Scalar]) -> NodalTemperatures {
    values
        .iter()
        .map(|&value| Quantity::<Temperature>::new(value))
        .collect()
}

pub(super) fn pseudorandom() -> Vec<Scalar> {
    (0..NODES)
        .map(|index| ((index as Scalar + 1.0) * 12.9898).sin() * 0.5 + 0.7)
        .collect()
}

mod heat {
    use super::*;
    use crate::{
        domain::{
            ElementModel, Model,
            thermal::{
                NodalForcesThermal, conduction::ThermalConductionElements,
                dynamics::ThermalConductionDynamics, time_scale::ThermalTimeScaleElements,
            },
            time_scale::{diffusive_time_scale_from_eigenvalue, largest_eigenvalue},
        },
        math::{Hessian, Tensor, integrate::Euler, optimize::EqualityConstraint},
        units::{Density, SpecificHeat, Time, Volume, VolumetricHeatCapacity},
        vem::block::element::VirtualElement,
    };
    type H = Block<Fourier, Element, Quantity<VolumetricHeatCapacity>>;
    fn capacity() -> Quantity<VolumetricHeatCapacity> {
        Density::kilograms_per_cubic_meter(7.8e3) * SpecificHeat::joules_per_kilogram_kelvin(450.0)
    }
    fn model() -> Model<H, 3> {
        (
            block().with_heat_capacity(capacity()),
            NodalReferenceCoordinates::from(coordinates()),
        )
            .into()
    }
    fn linear(gradient: [Scalar; 3]) -> Vec<Scalar> {
        coordinates()
            .iter()
            .map(|node| {
                300.0
                    + (0..3)
                        .map(|axis| gradient[axis] * node[axis])
                        .sum::<Scalar>()
            })
            .collect()
    }
    #[test]
    fn the_capacities_total_the_capacity_of_the_body() {
        let model = model();
        let total = model.nodal_lumped_heat_capacities().total();
        let volume = model
            .blocks
            .elements()
            .iter()
            .map(|element| element.integration_weights()[0])
            .sum::<Quantity<Volume>>();
        assert!(((total / (capacity() * volume)).value() - 1.0).abs() < 1e-12);
        assert!(
            model
                .nodal_lumped_heat_capacities()
                .iter()
                .all(|capacity| capacity.value() > 0.0)
        )
    }
    #[test]
    fn the_time_scale_bounds_the_global_one_from_above_and_is_not_loose() {
        let model = model();
        let state = temperatures(&linear([0.3, -0.7, 1.1]));
        let stiffnesses = model.nodal_stiffnesses(&state).unwrap();
        let capacities = model
            .nodal_lumped_heat_capacities()
            .iter()
            .map(|capacity| capacity.value())
            .collect::<Vec<_>>();
        let global = diffusive_time_scale_from_eigenvalue(largest_eigenvalue(
            NODES,
            |row, column| stiffnesses.entry(row, column),
            &capacities,
        ))
        .value();
        let elements = model
            .fastest_diffusive_time_scale(model.coordinates(), &state)
            .unwrap()
            .value();
        assert!(elements <= global * (1.0 + 1e-6), "{elements} > {global}");
        assert!(elements >= 0.2 * global, "{elements} << {global}")
    }
    #[test]
    fn a_perturbed_interior_node_relaxes_to_the_linear_field_with_the_boundary_held() {
        let model = model();
        let steady = linear([0.3, -0.7, 1.1]);
        let perturbation = 50.0;
        let mut initial = steady.clone();
        initial[INTERIOR] += perturbation;
        let initial = temperatures(&initial);
        let dt = 0.05
            * model
                .fastest_diffusive_time_scale(model.coordinates(), &initial)
                .unwrap()
                .value();
        let capacities = model.nodal_lumped_heat_capacities();
        let rate = model
            .nodal_stiffnesses(&initial)
            .unwrap()
            .entry(INTERIOR, INTERIOR)
            / capacities[INTERIOR].value();
        let steps = (8.0 / (rate * dt)).ceil() as usize;
        let times = (0..=steps)
            .map(|step| Time::seconds(step as Scalar * dt))
            .collect::<Vec<_>>();
        let boundary = (0..NODES).filter(|&node| node != INTERIOR).collect();
        let (_, history, rates) = model
            .integrate(
                &Euler::default(),
                &times,
                initial,
                &NodalForcesThermal::zero(NODES),
                &capacities,
                EqualityConstraint::Fixed(boundary),
            )
            .unwrap();
        history.iter().for_each(|temperatures| {
            (0..NODES)
                .filter(|&node| node != INTERIOR)
                .for_each(|node| assert_eq!(temperatures[node].value(), steady[node]))
        });
        rates.iter().for_each(|rates| {
            (0..NODES)
                .filter(|&node| node != INTERIOR)
                .for_each(|node| assert_eq!(rates[node].value(), 0.0))
        });
        let last = history.iter().last().unwrap();
        assert!((last[INTERIOR].value() - steady[INTERIOR]).abs() < 0.1);
        let middle = (2.0 / (rate * dt)).round() as usize;
        let exact = steady[INTERIOR] + perturbation * (-2.0_f64).exp();
        assert!(
            (history[middle][INTERIOR].value() - exact).abs() < 0.6,
            "{} != {exact}",
            history[middle][INTERIOR].value()
        )
    }
    mod bounded {
        use super::*;
        use crate::math::integrate::IntegrationError;
        const SAFETY: Scalar = 0.5;
        fn times(dt: Scalar, steps: usize) -> Vec<Quantity<Time>> {
            (0..=steps)
                .map(|step| Time::seconds(step as Scalar * dt))
                .collect()
        }
        fn step(model: &Model<H, 3>, state: &NodalTemperatures) -> Scalar {
            0.4 * model
                .fastest_diffusive_time_scale(model.coordinates(), state)
                .unwrap()
                .value()
        }
        #[test]
        fn an_insulated_body_conserves_its_heat_and_relaxes_to_the_mean() {
            let model = model();
            let initial = temperatures(
                &pseudorandom()
                    .iter()
                    .map(|value| 300.0 + 100.0 * value)
                    .collect::<Vec<_>>(),
            );
            let dt = step(&model, &initial);
            let capacities = model.nodal_lumped_heat_capacities();
            let heat = |temperatures: &NodalTemperatures| -> Scalar {
                capacities
                    .iter()
                    .zip(temperatures.iter())
                    .map(|(capacity, temperature)| capacity.value() * temperature.value())
                    .sum()
            };
            let (_, history, _) = model
                .integrate_bounded(
                    &Euler::default(),
                    SAFETY,
                    1,
                    &times(dt, 20000),
                    initial.clone(),
                    &NodalForcesThermal::zero(NODES),
                    &capacities,
                    EqualityConstraint::None,
                )
                .unwrap();
            let last = history.iter().last().unwrap();
            let total = heat(&initial);
            assert!((heat(last) - total).abs() <= 1e-9 * total.abs());
            let mean = total
                / capacities
                    .iter()
                    .map(|capacity| capacity.value())
                    .sum::<Scalar>();
            last.iter()
                .for_each(|temperature| assert!((temperature.value() - mean).abs() < 1e-3 * mean))
        }
        #[test]
        fn a_held_boundary_is_kept_and_the_interior_relaxes_to_the_linear_field() {
            let model = model();
            let steady = linear([0.3, -0.7, 1.1]);
            let mut initial = steady.clone();
            initial[INTERIOR] += 50.0;
            let initial = temperatures(&initial);
            let dt = step(&model, &initial);
            let boundary: Vec<usize> = (0..NODES).filter(|&node| node != INTERIOR).collect();
            let (_, history, rates) = model
                .integrate_bounded(
                    &Euler::default(),
                    SAFETY,
                    1,
                    &times(dt, 20000),
                    initial,
                    &NodalForcesThermal::zero(NODES),
                    &model.nodal_lumped_heat_capacities(),
                    EqualityConstraint::Fixed(boundary.clone()),
                )
                .unwrap();
            history.iter().for_each(|temperatures| {
                boundary
                    .iter()
                    .for_each(|&node| assert_eq!(temperatures[node].value(), steady[node]))
            });
            rates.iter().for_each(|rates| {
                boundary
                    .iter()
                    .for_each(|&node| assert_eq!(rates[node].value(), 0.0))
            });
            let last = history.iter().last().unwrap();
            assert!((last[INTERIOR].value() - steady[INTERIOR]).abs() < 1e-3)
        }
        #[test]
        fn a_step_above_the_limit_is_an_error_and_would_diverge() {
            let model = model();
            let initial = temperatures(&linear([0.3, -0.7, 1.1]));
            let dt = 3.0
                * model
                    .fastest_diffusive_time_scale(model.coordinates(), &initial)
                    .unwrap()
                    .value();
            let mut perturbed = linear([0.3, -0.7, 1.1]);
            perturbed[INTERIOR] += 50.0;
            let perturbed = temperatures(&perturbed);
            let capacities = model.nodal_lumped_heat_capacities();
            let error = model
                .integrate_bounded(
                    &Euler::default(),
                    1.0,
                    1,
                    &times(dt, 40),
                    perturbed.clone(),
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
                    perturbed,
                    &NodalForcesThermal::zero(NODES),
                    &capacities,
                    EqualityConstraint::None,
                )
                .unwrap();
            assert!(
                history
                    .iter()
                    .last()
                    .unwrap()
                    .iter()
                    .any(|temperature| temperature.value().abs() > 1e6)
            )
        }
        #[test]
        fn an_interval_of_zero_is_an_error() {
            let model = model();
            let initial = temperatures(&linear([0.3, -0.7, 1.1]));
            let error = model
                .integrate_bounded(
                    &Euler::default(),
                    SAFETY,
                    0,
                    &times(1.0, 2),
                    initial,
                    &NodalForcesThermal::zero(NODES),
                    &model.nodal_lumped_heat_capacities(),
                    EqualityConstraint::None,
                )
                .unwrap_err();
            assert!(matches!(error, IntegrationError::Intermediate(..)))
        }
    }
}
