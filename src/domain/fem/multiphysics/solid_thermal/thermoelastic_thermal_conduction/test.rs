use crate::{
    constitutive::{
        multiphysics::Canonical,
        solid::thermoelastic::{
            AlmansiHamel,
            test::{
                BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE,
                SHEAR_MODULUS,
            },
        },
        thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
    },
    fem::{
        Model, NodalReferenceCoordinates, Root,
        block::{Block, element::linear::Tetrahedron},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
        },
    },
    math::{
        Matrix, Vector,
        optimize::{EqualityConstraint, NewtonRaphson},
    },
};

const NODES: usize = 5;
const CONNECTIVITY: [[usize; 4]; 2] = [[0, 1, 2, 3], [1, 2, 3, 4]];
const HOT: f64 = 100.02;

type B = Block<Canonical<AlmansiHamel, Fourier>, Tetrahedron<4>, 4, 3, 4, 4>;

fn coordinates() -> NodalReferenceCoordinates<3> {
    [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ]
    .into()
}

fn model() -> Model<B, 3> {
    (
        (
            (
                AlmansiHamel {
                    bulk_modulus: BULK_MODULUS,
                    shear_modulus: SHEAR_MODULUS,
                    coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                    reference_temperature: REFERENCE_TEMPERATURE,
                },
                Fourier {
                    thermal_conductivity: THERMAL_CONDUCTIVITY,
                },
            )
                .into(),
            CONNECTIVITY.to_vec(),
            &coordinates(),
        )
            .into(),
        coordinates(),
    )
        .into()
}

fn dilation() -> f64 {
    let temperature_change = HOT - REFERENCE_TEMPERATURE.value();
    (1.0 - 2.0 * COEFFICIENT_OF_THERMAL_EXPANSION.value() * temperature_change).powf(-0.5)
}

fn constraint() -> EqualityConstraint {
    let dilation = dilation();
    let mut matrix = Matrix::zero(11, 4 * NODES);
    let mut rhs = Vector::zero(11);
    (0..NODES).for_each(|node| {
        matrix[node][4 * node + 3] = 1.0;
        rhs[node] = HOT
    });
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        .iter()
        .enumerate()
        .for_each(|(row, &(node, direction))| {
            matrix[NODES + row][4 * node + direction] = 1.0;
            rhs[NODES + row] = dilation * coordinates()[node][direction].value()
        });
    EqualityConstraint::Linear(matrix, rhs)
}

fn fixed() -> EqualityConstraint {
    EqualityConstraint::Fixed(
        (0..NODES)
            .map(|node| 4 * node + 3)
            .chain([0, 1, 2, 5, 6, 10])
            .collect(),
    )
}

fn solve(constraint: EqualityConstraint) -> NodalStatesSolidThermal {
    <Model<B, 3> as Root<
        NodalForcesSolidThermal,
        NodalStiffnessesSolidThermal,
        NodalStatesSolidThermal,
    >>::root(&model(), constraint, NewtonRaphson::default())
    .unwrap()
}

#[test]
fn a_uniform_temperature_dilates_the_body_without_stress() {
    let dilation = dilation();
    let states = solve(constraint());
    (0..NODES).for_each(|node| {
        (0..3).for_each(|direction| {
            let expected = dilation * coordinates()[node][direction].value();
            let found = states[node].0[direction].value();
            assert!(
                (found - expected).abs() < 1e-8,
                "node {node} direction {direction}: {found} != {expected}"
            )
        })
    })
}

#[test]
fn the_imposed_temperature_is_reached_at_every_node() {
    let states = solve(constraint());
    (0..NODES).for_each(|node| {
        let found = states[node].1.value();
        assert!((found - HOT).abs() < 1e-8, "node {node}: {found} != {HOT}")
    })
}

#[test]
fn fixed_temperatures_hold_the_reference_temperature_and_leave_the_body_unstrained() {
    let states = solve(fixed());
    (0..NODES).for_each(|node| {
        (0..3).for_each(|direction| {
            let expected = coordinates()[node][direction].value();
            let found = states[node].0[direction].value();
            assert!(
                (found - expected).abs() < 1e-10,
                "node {node} direction {direction}: {found} != {expected}"
            )
        });
        let found = states[node].1.value();
        let expected = REFERENCE_TEMPERATURE.value();
        assert!(
            (found - expected).abs() < 1e-10,
            "node {node}: {found} != {expected}"
        )
    })
}

mod dynamics {
    use crate::{
        constitutive::{
            multiphysics::Canonical,
            solid::thermoelastic::{
                AlmansiHamel,
                test::{
                    BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION, REFERENCE_TEMPERATURE,
                    SHEAR_MODULUS,
                },
            },
            thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
        },
        fem::{
            Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
            block::{
                Block,
                element::linear::Tetrahedron,
                thermal::{NodalTemperatures, conduction::NodalForcesThermal},
            },
            multiphysics::solid_thermal::thermoelastic_thermal_conduction::dynamics::{
                ThermoelasticThermalConductionDynamics,
                time_scale::ThermoelasticThermalConductionTimeScaleElements,
            },
            solid::NodalForcesSolid,
        },
        math::{
            Quantity, Tensor,
            assert::{Assert, AssertionError},
            integrate::{
                ButcherTableau, DormandPrinceFixedStep, DormandPrinceTableau, IntegrationError,
            },
            optimize::EqualityConstraint,
        },
        units::{Density, Time, VolumetricHeatCapacity},
    };
    use std::f64::consts::FRAC_PI_2;
    const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);
    const HEAT_CAPACITY: f64 = 3.0e6;
    const STRETCH_RATE: f64 = 0.01;
    const COORDINATES: [[f64; 3]; 5] = [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
        [1.5, 1.4, 1.3],
    ];
    type Coupled = Canonical<AlmansiHamel, Fourier>;
    type Massive = Block<Coupled, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;
    type Both = Block<
        Coupled,
        Tetrahedron<4>,
        4,
        3,
        4,
        4,
        Quantity<Density>,
        Quantity<VolumetricHeatCapacity>,
    >;
    fn coupled() -> Coupled {
        (
            AlmansiHamel {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
                coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                reference_temperature: REFERENCE_TEMPERATURE,
            },
            Fourier {
                thermal_conductivity: THERMAL_CONDUCTIVITY,
            },
        )
            .into()
    }
    fn connectivity() -> Vec<[usize; 4]> {
        vec![[0, 1, 2, 3], [1, 2, 3, 4]]
    }
    fn reference() -> NodalReferenceCoordinates<3> {
        COORDINATES.into()
    }
    fn model() -> Model<Both, 3> {
        (
            Massive::from((coupled(), DENSITY, connectivity(), &reference()))
                .with_heat_capacity(Quantity::<VolumetricHeatCapacity>::new(HEAT_CAPACITY)),
            reference(),
        )
            .into()
    }
    fn uniform_temperatures() -> NodalTemperatures {
        COORDINATES
            .map(|_| REFERENCE_TEMPERATURE)
            .into_iter()
            .collect()
    }
    fn stretching() -> NodalVelocities<3> {
        COORDINATES
            .map(|coordinate| coordinate.map(|component| STRETCH_RATE * component))
            .into()
    }
    fn translating() -> NodalVelocities<3> {
        COORDINATES.map(|_| [0.3, -0.2, 0.1]).into()
    }
    fn time() -> Vec<Quantity<Time>> {
        (0..=10)
            .map(|step| Time::seconds(1e-5 * step as f64))
            .collect()
    }
    fn no_forces() -> NodalForcesSolid<3> {
        NodalForcesSolid::zero(COORDINATES.len())
    }
    fn no_heating() -> NodalForcesThermal {
        NodalForcesThermal::zero(COORDINATES.len())
    }
    fn integrate(
        velocities: NodalVelocities<3>,
        equality_constraint: EqualityConstraint,
    ) -> (NodalCoordinates<3>, NodalTemperatures) {
        let (_, coordinates, _, _, temperatures, _) = model()
            .integrate(
                &DormandPrinceFixedStep::default(),
                &time(),
                (NodalCoordinates::from(COORDINATES), velocities),
                uniform_temperatures(),
                &no_forces(),
                &no_heating(),
                &model().nodal_lumped_masses(),
                &model().nodal_lumped_heat_capacities(),
                equality_constraint,
            )
            .unwrap();
        (
            coordinates[coordinates.len() - 1].clone(),
            temperatures[temperatures.len() - 1].clone(),
        )
    }
    #[test]
    fn stretching_cools_at_the_adiabatic_rate() -> Result<(), AssertionError> {
        let (accelerations, rates) = model()
            .nodal_rates(
                &NodalCoordinates::from(COORDINATES),
                &stretching(),
                &uniform_temperatures(),
                &no_forces(),
                &no_heating(),
                &model().nodal_lumped_masses(),
                &model().nodal_lumped_heat_capacities(),
            )
            .unwrap();
        let expected = -9.0
            * COEFFICIENT_OF_THERMAL_EXPANSION.value()
            * BULK_MODULUS.value()
            * STRETCH_RATE
            * REFERENCE_TEMPERATURE.value()
            / HEAT_CAPACITY;
        assert!(expected < 0.0);
        rates
            .iter()
            .try_for_each(|rate| Assert::default().eq_within_tols(rate.value(), &expected))?;
        accelerations.iter().for_each(|acceleration| {
            (0..3).for_each(|i| assert!(acceleration[i].value().abs() < 1e-9))
        });
        Ok(())
    }
    #[test]
    fn a_rigid_translation_neither_heats_nor_cools() -> Result<(), AssertionError> {
        let (coordinates, temperatures) = integrate(translating(), EqualityConstraint::None);
        temperatures.iter().try_for_each(|temperature| {
            Assert::default().eq_within_tols(temperature, &REFERENCE_TEMPERATURE)
        })?;
        coordinates
            .iter()
            .zip(COORDINATES)
            .try_for_each(|(coordinate, initial)| {
                [0.3, -0.2, 0.1]
                    .iter()
                    .enumerate()
                    .try_for_each(|(i, velocity)| {
                        Assert::default()
                            .eq_within_tols(coordinate[i].value(), &(initial[i] + velocity * 1e-4))
                    })
            })
    }
    #[test]
    fn a_stretching_body_cools() {
        let (_, temperatures) = integrate(stretching(), EqualityConstraint::None);
        temperatures
            .iter()
            .for_each(|temperature| assert!(temperature.value() < REFERENCE_TEMPERATURE.value()));
    }
    #[test]
    fn a_fixed_temperature_is_held() -> Result<(), AssertionError> {
        let (_, temperatures) = integrate(stretching(), EqualityConstraint::Fixed(vec![3]));
        Assert::default().eq_within_tols(temperatures[0], &REFERENCE_TEMPERATURE)?;
        assert!(temperatures[1].value() < REFERENCE_TEMPERATURE.value());
        Ok(())
    }
    #[test]
    fn a_fixed_coordinate_is_held() -> Result<(), AssertionError> {
        let (coordinates, _) = integrate(stretching(), EqualityConstraint::Fixed(vec![5]));
        Assert::default().eq_within_tols(coordinates[1][1].value(), &COORDINATES[1][1])?;
        assert!((coordinates[1][0].value() - COORDINATES[1][0]).abs() > 0.0);
        Ok(())
    }
    #[test]
    fn linear_constraints_are_rejected() {
        use crate::math::{Matrix, Vector};
        let result = model().integrate(
            &DormandPrinceFixedStep::default(),
            &time(),
            (NodalCoordinates::from(COORDINATES), stretching()),
            uniform_temperatures(),
            &no_forces(),
            &no_heating(),
            &model().nodal_lumped_masses(),
            &model().nodal_lumped_heat_capacities(),
            EqualityConstraint::Linear(Matrix::zero(1, 20), Vector::zero(1)),
        );
        assert!(result.is_err());
    }
    fn integrate_bounded(
        time: &[Quantity<Time>],
        safety: f64,
        interval: usize,
    ) -> Result<(NodalCoordinates<3>, NodalTemperatures), IntegrationError> {
        let (_, coordinates, _, _, temperatures, _) = model().integrate_bounded(
            &DormandPrinceFixedStep::default(),
            safety,
            interval,
            time,
            (NodalCoordinates::from(COORDINATES), stretching()),
            uniform_temperatures(),
            &no_forces(),
            &no_heating(),
            &model().nodal_lumped_masses(),
            &model().nodal_lumped_heat_capacities(),
            EqualityConstraint::None,
        )?;
        Ok((
            coordinates[coordinates.len() - 1].clone(),
            temperatures[temperatures.len() - 1].clone(),
        ))
    }
    #[test]
    fn a_stable_step_agrees_with_the_unbounded_integration() -> Result<(), AssertionError> {
        let (coordinates, temperatures) = integrate_bounded(&time(), 0.9, 1).unwrap();
        let (expected_coordinates, expected_temperatures) =
            integrate(stretching(), EqualityConstraint::None);
        Assert::default().eq_within_tols(&coordinates, &expected_coordinates)?;
        Assert::default().eq_within_tols(&temperatures, &expected_temperatures)
    }
    #[test]
    fn an_unstable_step_is_an_error() {
        let (wave, diffusion) = model()
            .fastest_time_scales(
                &reference(),
                &NodalCoordinates::from(COORDINATES),
                &uniform_temperatures(),
            )
            .unwrap();
        let step = 1.0e3 * wave.value().min(diffusion.value());
        let result = integrate_bounded(
            &[
                Time::seconds(0.0),
                Time::seconds(step),
                Time::seconds(2.0 * step),
            ],
            1.0,
            1,
        )
        .map(|_| ());
        let stability = DormandPrinceTableau::stability();
        let expected = (diffusion.value() * stability.extent(0.0))
            .min(wave.value() * stability.extent(FRAC_PI_2));
        match result {
            Err(IntegrationError::UnstableTimeStep(_, limit, _)) => {
                assert!((limit - expected).abs() < 1e-9 * expected)
            }
            _ => panic!("expected an unstable time step: {result:?}"),
        }
    }
    #[test]
    fn a_refresh_interval_of_zero_is_an_error() {
        assert!(integrate_bounded(&time(), 0.9, 0).is_err());
    }
    const BAR_ELEMENTS: usize = 10;
    const BAR_LENGTH: f64 = 1.0;
    const BAR_WIDTH: f64 = 0.1;
    const BAR_HEAT_CAPACITY: f64 = 3.0e4;
    const BAR_AMPLITUDE: f64 = 1.0e-6;
    fn bar_node(i: usize, j: usize, k: usize) -> usize {
        i + (BAR_ELEMENTS + 1) * (j + 2 * k)
    }
    fn bar_coordinates() -> Vec<[f64; 3]> {
        let spacing = BAR_LENGTH / BAR_ELEMENTS as f64;
        (0..2)
            .flat_map(|k| {
                (0..2).flat_map(move |j| {
                    (0..=BAR_ELEMENTS).map(move |i| {
                        [
                            spacing * i as f64,
                            BAR_WIDTH * j as f64,
                            BAR_WIDTH * k as f64,
                        ]
                    })
                })
            })
            .collect()
    }
    fn bar_connectivity() -> Vec<[usize; 4]> {
        let coordinates = bar_coordinates();
        (0..BAR_ELEMENTS)
            .flat_map(|i| {
                let vertex = move |bit: usize| bar_node(i + (bit & 1), (bit >> 1) & 1, bit >> 2);
                [
                    [0, 1, 3, 7],
                    [0, 1, 5, 7],
                    [0, 2, 3, 7],
                    [0, 2, 6, 7],
                    [0, 4, 5, 7],
                    [0, 4, 6, 7],
                ]
                .map(move |tetrahedron| tetrahedron.map(vertex))
            })
            .map(|mut nodes| {
                let edge = |a: usize, b: usize| {
                    [0, 1, 2].map(|i| coordinates[nodes[b]][i] - coordinates[nodes[a]][i])
                };
                let (a, b, c) = (edge(0, 1), edge(0, 2), edge(0, 3));
                let volume = a[0] * (b[1] * c[2] - b[2] * c[1])
                    - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                if volume < 0.0 {
                    nodes.swap(2, 3);
                }
                nodes
            })
            .collect()
    }
    fn bar(heat_capacity: f64) -> Model<Both, 3> {
        let reference = NodalReferenceCoordinates::from(bar_coordinates());
        (
            Massive::from((coupled(), DENSITY, bar_connectivity(), &reference))
                .with_heat_capacity(Quantity::<VolumetricHeatCapacity>::new(heat_capacity)),
            reference,
        )
            .into()
    }
    fn bar_constraint() -> EqualityConstraint {
        EqualityConstraint::Fixed(
            (0..4 * (BAR_ELEMENTS + 1))
                .flat_map(|node| {
                    let clamped = (node % (BAR_ELEMENTS + 1) == 0).then_some(4 * node);
                    [4 * node + 1, 4 * node + 2].into_iter().chain(clamped)
                })
                .collect(),
        )
    }
    fn bar_velocities() -> NodalVelocities<3> {
        bar_coordinates()
            .iter()
            .map(|coordinate| {
                [
                    BAR_AMPLITUDE * (FRAC_PI_2 * coordinate[0] / BAR_LENGTH).sin(),
                    0.0,
                    0.0,
                ]
            })
            .collect::<Vec<_>>()
            .into()
    }
    fn bar_quarter_period(heat_capacity: f64) -> f64 {
        let model = bar(heat_capacity);
        let nodes = 4 * (BAR_ELEMENTS + 1);
        let time = (0..=800)
            .map(|step| Time::seconds(0.05 * step as f64))
            .collect::<Vec<_>>();
        let (times, _, velocities, ..) = model
            .integrate_bounded(
                &DormandPrinceFixedStep::default(),
                0.5,
                1,
                &time,
                (NodalCoordinates::from(bar_coordinates()), bar_velocities()),
                (0..nodes).map(|_| REFERENCE_TEMPERATURE).collect(),
                &NodalForcesSolid::zero(nodes),
                &NodalForcesThermal::zero(nodes),
                &model.nodal_lumped_masses(),
                &model.nodal_lumped_heat_capacities(),
                bar_constraint(),
            )
            .unwrap();
        let tip = bar_node(BAR_ELEMENTS, 0, 0);
        let tip_velocities = velocities
            .iter()
            .map(|velocity| velocity[tip][0].value())
            .collect::<Vec<_>>();
        let step = (1..tip_velocities.len())
            .find(|&step| tip_velocities[step] <= 0.0)
            .expect("the tip velocity must change sign within the simulated time");
        let fraction = tip_velocities[step - 1] / (tip_velocities[step - 1] - tip_velocities[step]);
        times[step - 1].value() + fraction * (times[step] - times[step - 1]).value()
    }
    #[test]
    fn the_isothermal_wave_travels_at_the_uniaxial_strain_modulus() -> Result<(), AssertionError> {
        let modulus = BULK_MODULUS.value() + 4.0 / 3.0 * SHEAR_MODULUS.value();
        let expected = BAR_LENGTH * (DENSITY.value() / modulus).sqrt();
        Assert {
            rel_tol: 1e-2,
            ..Assert::default()
        }
        .eq_within_tols(bar_quarter_period(1.0e6 * BAR_HEAT_CAPACITY), &expected)
    }
    #[test]
    fn the_adiabatic_wave_is_faster_by_the_thermoelastic_stiffening() -> Result<(), AssertionError>
    {
        let bulk_modulus = BULK_MODULUS.value();
        let modulus = bulk_modulus + 4.0 / 3.0 * SHEAR_MODULUS.value();
        let stiffening = 9.0
            * bulk_modulus
            * bulk_modulus
            * COEFFICIENT_OF_THERMAL_EXPANSION.value().powi(2)
            * REFERENCE_TEMPERATURE.value()
            / (BAR_HEAT_CAPACITY * modulus);
        assert!(stiffening > 0.1);
        let ratio =
            bar_quarter_period(1.0e6 * BAR_HEAT_CAPACITY) / bar_quarter_period(BAR_HEAT_CAPACITY);
        assert!(ratio > 1.0);
        Assert {
            rel_tol: 1e-2,
            ..Assert::default()
        }
        .eq_within_tols(ratio, &(1.0 + stiffening).sqrt())
    }
    use crate::{
        constitutive::solid::thermohyperelastic::NeoHookean, units::ReciprocalTemperature,
    };
    type Entropic = Canonical<NeoHookean, Fourier>;
    type EntropicMassive = Block<Entropic, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;
    type EntropicBoth = Block<
        Entropic,
        Tetrahedron<4>,
        4,
        3,
        4,
        4,
        Quantity<Density>,
        Quantity<VolumetricHeatCapacity>,
    >;
    fn entropic_model() -> Model<EntropicBoth, 3> {
        (
            EntropicMassive::from((
                (
                    NeoHookean {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                        coefficient_of_thermal_expansion: ReciprocalTemperature::per_kelvin(0.0),
                        reference_temperature: REFERENCE_TEMPERATURE,
                    },
                    Fourier {
                        thermal_conductivity: THERMAL_CONDUCTIVITY,
                    },
                )
                    .into(),
                DENSITY,
                connectivity(),
                &reference(),
            ))
            .with_heat_capacity(Quantity::<VolumetricHeatCapacity>::new(HEAT_CAPACITY)),
            reference(),
        )
            .into()
    }
    #[test]
    fn an_entropic_solid_warms_at_the_rate_of_the_shear_work() {
        let stretch: f64 = 1.3;
        let stretches = [stretch, stretch.powf(-0.5), stretch.powf(-0.5)];
        let directions = [1.0, -0.5, -0.5];
        let coordinates =
            NodalCoordinates::from(COORDINATES.map(|x| [0, 1, 2].map(|i| stretches[i] * x[i])));
        let velocities = NodalVelocities::from(
            COORDINATES
                .map(|x| [0, 1, 2].map(|i| STRETCH_RATE * directions[i] * stretches[i] * x[i])),
        );
        let model = entropic_model();
        let (_, rates) = model
            .nodal_rates(
                &coordinates,
                &velocities,
                &uniform_temperatures(),
                &no_forces(),
                &no_heating(),
                &model.nodal_lumped_masses(),
                &model.nodal_lumped_heat_capacities(),
            )
            .unwrap();
        let squares = stretches.map(|stretch| stretch * stretch);
        let mean = squares.iter().sum::<f64>() / 3.0;
        let work = (0..3)
            .map(|i| (squares[i] - mean) * directions[i])
            .sum::<f64>()
            * SHEAR_MODULUS.value()
            * STRETCH_RATE;
        let expected = work / HEAT_CAPACITY;
        assert!(expected > 0.0);
        rates.iter().for_each(|rate| {
            assert!(
                (rate.value() - expected).abs() < 1e-6 * expected,
                "{} != {expected}",
                rate.value()
            )
        });
    }
}
