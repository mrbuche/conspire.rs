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
        boundary::{BoundaryConditions, Displacement, SolidThermal, Temperature},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
        },
    },
    math::{
        Matrix, Quantity, Vector,
        optimize::{EqualityConstraint, EqualityConstraints, NewtonRaphson},
    },
    units::{Length, Temperature as TemperatureUnit},
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

fn constraint() -> EqualityConstraints {
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
    vec![EqualityConstraint::Linear(matrix, rhs)]
}

fn prescribed_conditions() -> BoundaryConditions<'static, SolidThermal<3>> {
    let moved = |node: usize, direction: usize| {
        Quantity::<Length>::new((dilation() - 1.0) * coordinates()[node][direction].value())
    };
    BoundaryConditions::new()
        .prescribe_nodes(Temperature, 0..NODES, Quantity::<TemperatureUnit>::new(HOT))
        .prescribe_node(Displacement::<0>, 0, moved(0, 0))
        .prescribe_node(Displacement::<1>, 0, moved(0, 1))
        .prescribe_node(Displacement::<2>, 0, moved(0, 2))
        .prescribe_node(Displacement::<1>, 1, moved(1, 1))
        .prescribe_node(Displacement::<2>, 1, moved(1, 2))
        .prescribe_node(Displacement::<2>, 2, moved(2, 2))
}

fn fixed_conditions() -> BoundaryConditions<'static, SolidThermal<3>> {
    BoundaryConditions::new()
        .fix_nodes(Temperature, 0..NODES)
        .fix_node(Displacement::<0>, 0)
        .fix_node(Displacement::<1>, 0)
        .fix_node(Displacement::<2>, 0)
        .fix_node(Displacement::<1>, 1)
        .fix_node(Displacement::<2>, 1)
        .fix_node(Displacement::<2>, 2)
}

fn fixed() -> EqualityConstraints {
    fixed_conditions().constraint()
}

fn solve(constraint: EqualityConstraints) -> NodalStatesSolidThermal {
    <Model<B, 3> as Root<
        NodalForcesSolidThermal,
        NodalStiffnessesSolidThermal,
        NodalStatesSolidThermal,
    >>::root(&model(), constraint, NewtonRaphson::default())
    .unwrap()
}

#[test]
fn boundary_conditions_of_the_layout_of_the_state_give_the_same_solution() {
    let states = <Model<B, 3> as Root<
        NodalForcesSolidThermal,
        NodalStiffnessesSolidThermal,
        NodalStatesSolidThermal,
    >>::root_with(&model(), &fixed_conditions(), NewtonRaphson::default())
    .unwrap();
    let expected = solve(fixed());
    (0..NODES).for_each(|node| {
        (0..3).for_each(|direction| {
            assert_eq!(
                states[node].0[direction].value(),
                expected[node].0[direction].value()
            )
        });
        assert_eq!(states[node].1.value(), expected[node].1.value())
    })
}

#[test]
fn prescribed_values_found_by_elimination_are_what_multipliers_find() {
    let dilation = dilation();
    let eliminated = <Model<B, 3> as Root<
        NodalForcesSolidThermal,
        NodalStiffnessesSolidThermal,
        NodalStatesSolidThermal,
    >>::root_with(&model(), &prescribed_conditions(), NewtonRaphson::default())
    .unwrap();
    let multiplied = solve(constraint());
    (0..NODES).for_each(|node| {
        (0..3).for_each(|direction| {
            let found = eliminated[node].0[direction].value();
            let expected = dilation * coordinates()[node][direction].value();
            assert!(
                (found - expected).abs() < 1e-8,
                "node {node} direction {direction}: {found} != {expected}"
            );
            assert!(
                (found - multiplied[node].0[direction].value()).abs() < 1e-8,
                "node {node} direction {direction}: {found} != multipliers"
            )
        });
        let found = eliminated[node].1.value();
        assert!((found - HOT).abs() < 1e-8, "node {node}: {found} != {HOT}");
        assert!((found - multiplied[node].1.value()).abs() < 1e-8)
    })
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
            optimize::{EqualityConstraint, EqualityConstraints},
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
        equality_constraint: EqualityConstraints,
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
        let (coordinates, temperatures) = integrate(translating(), vec![EqualityConstraint::None]);
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
        let (_, temperatures) = integrate(stretching(), vec![EqualityConstraint::None]);
        temperatures
            .iter()
            .for_each(|temperature| assert!(temperature.value() < REFERENCE_TEMPERATURE.value()));
    }
    #[test]
    fn a_fixed_temperature_is_held() -> Result<(), AssertionError> {
        let (_, temperatures) = integrate(stretching(), vec![EqualityConstraint::Fixed(vec![3])]);
        Assert::default().eq_within_tols(temperatures[0], &REFERENCE_TEMPERATURE)?;
        assert!(temperatures[1].value() < REFERENCE_TEMPERATURE.value());
        Ok(())
    }
    #[test]
    fn a_fixed_coordinate_is_held() -> Result<(), AssertionError> {
        let (coordinates, _) = integrate(stretching(), vec![EqualityConstraint::Fixed(vec![5])]);
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
            vec![EqualityConstraint::Linear(
                Matrix::zero(1, 20),
                Vector::zero(1),
            )],
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
            vec![EqualityConstraint::None],
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
            integrate(stretching(), vec![EqualityConstraint::None]);
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
                vec![bar_constraint()],
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
                        coefficient_of_thermal_expansion_slope: Quantity::new(0.0),
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
    mod implicit {
        use super::*;
        use crate::{
            fem::multiphysics::solid_thermal::thermoelastic_thermal_conduction::dynamics::implicit::ThermoelasticThermalConductionImplicitDynamics,
            math::{
                integrate::Newmark,
                optimize::{NewtonRaphson, Tolerances},
            },
        };
        fn solver() -> NewtonRaphson {
            NewtonRaphson {
                abs_tol: Tolerances {
                    constraint: 1e-12,
                    residual: 1e-3,
                },
                rel_tol: Some(1e-10),
                ..Default::default()
            }
        }
        fn implicit(
            velocities: NodalVelocities<3>,
            equality_constraint: EqualityConstraints,
            time: &[Quantity<Time>],
        ) -> Result<(NodalCoordinates<3>, NodalTemperatures), IntegrationError> {
            let model = model();
            let (_, coordinates, _, _, temperatures, _) =
                ThermoelasticThermalConductionImplicitDynamics::integrate(
                    &model,
                    &Newmark::default(),
                    solver(),
                    time,
                    (NodalCoordinates::from(COORDINATES), velocities),
                    uniform_temperatures(),
                    &no_forces(),
                    &no_heating(),
                    &model.nodal_lumped_masses(),
                    &model.nodal_lumped_heat_capacities(),
                    equality_constraint,
                )?;
            Ok((
                coordinates[coordinates.len() - 1].clone(),
                temperatures[temperatures.len() - 1].clone(),
            ))
        }
        #[test]
        fn a_stretching_body_cools_like_the_explicit_integration() {
            let (coordinates, temperatures) =
                implicit(stretching(), vec![EqualityConstraint::None], &time()).unwrap();
            let (explicit_coordinates, explicit_temperatures) =
                integrate(stretching(), vec![EqualityConstraint::None]);
            temperatures
                .iter()
                .zip(explicit_temperatures.iter())
                .for_each(|(found, expected)| {
                    let change = expected.value() - REFERENCE_TEMPERATURE.value();
                    assert!(change < 0.0);
                    assert!(
                        (found.value() - expected.value()).abs() < 1e-3 * change.abs(),
                        "{} != {}",
                        found.value(),
                        expected.value()
                    )
                });
            coordinates
                .iter()
                .zip(explicit_coordinates.iter())
                .zip(COORDINATES)
                .for_each(|((found, expected), initial)| {
                    (0..3).for_each(|i| {
                        let change = expected[i].value() - initial[i];
                        assert!(
                            (found[i].value() - expected[i].value()).abs()
                                < 1e-3 * change.abs().max(1e-12)
                        )
                    })
                });
        }
        #[test]
        fn the_velocity_rate_scales_the_heating_into_the_thermal_row() {
            use crate::{
                fem::multiphysics::solid_thermal::thermoelastic_thermal_conduction::{
                    rates::ThermoelasticThermalConductionRateElements, rates::add_heatings,
                },
                math::Hessian,
                units::Rate,
            };
            let rate = 7.0;
            let (mut tangent, heatings) =
                ThermoelasticThermalConductionRateElements::nodal_stiffnesses(
                    &model(),
                    &NodalCoordinates::from(COORDINATES),
                    &stretching(),
                    &uniform_temperatures(),
                )
                .unwrap();
            let before = (0..COORDINATES.len())
                .map(|a| {
                    (0..COORDINATES.len())
                        .map(|b| {
                            (0..3)
                                .map(|k| tangent.entry(4 * a + 3, 4 * b + k))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            add_heatings(&mut tangent, &heatings, Quantity::<Rate>::new(rate));
            let mut any = false;
            (0..COORDINATES.len()).for_each(|a| {
                (0..COORDINATES.len()).for_each(|b| {
                    (0..3).for_each(|k| {
                        let heating = heatings[a]
                            .entries()
                            .find(|&(column, _)| column == b)
                            .map_or(0.0, |(_, heating)| heating[k].value());
                        let expected = rate * heating;
                        any |= expected != 0.0;
                        assert!(
                            (tangent.entry(4 * a + 3, 4 * b + k) - before[a][b][k] - expected)
                                .abs()
                                <= 1e-12 * expected.abs()
                        );
                    })
                })
            });
            assert!(any);
        }
        #[test]
        fn a_rigid_translation_neither_heats_nor_cools() {
            let (coordinates, temperatures) =
                implicit(translating(), vec![EqualityConstraint::None], &time()).unwrap();
            temperatures.iter().for_each(|temperature| {
                assert!((temperature.value() - REFERENCE_TEMPERATURE.value()).abs() < 1e-9)
            });
            coordinates
                .iter()
                .zip(COORDINATES)
                .for_each(|(coordinate, initial)| {
                    [0.3, -0.2, 0.1]
                        .iter()
                        .enumerate()
                        .for_each(|(i, velocity)| {
                            assert!(
                                (coordinate[i].value() - (initial[i] + velocity * 1e-4)).abs()
                                    < 1e-9
                            )
                        })
                });
        }
        #[test]
        fn a_fixed_temperature_is_held() {
            let (_, temperatures) = implicit(
                stretching(),
                vec![EqualityConstraint::Fixed(vec![3])],
                &time(),
            )
            .unwrap();
            assert_eq!(temperatures[0].value(), REFERENCE_TEMPERATURE.value());
            assert!(temperatures[1].value() < REFERENCE_TEMPERATURE.value());
        }
        #[test]
        fn a_fixed_coordinate_is_held() {
            let (coordinates, _) = implicit(
                stretching(),
                vec![EqualityConstraint::Fixed(vec![0])],
                &time(),
            )
            .unwrap();
            assert_eq!(coordinates[0][0].value(), COORDINATES[0][0]);
        }
        #[test]
        fn linear_constraints_are_rejected() {
            assert!(
                implicit(
                    stretching(),
                    vec![EqualityConstraint::Linear(
                        crate::math::Matrix::zero(1, 20),
                        crate::math::Vector::zero(1)
                    )],
                    &time()
                )
                .is_err()
            );
        }
    }
}

mod quasistatic {
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
            Model, NodalCoordinates, NodalReferenceCoordinates,
            block::{
                Block,
                element::linear::Tetrahedron,
                thermal::{NodalTemperatures, conduction::NodalForcesThermal},
            },
            multiphysics::solid_thermal::thermoelastic_thermal_conduction::quasistatic::ThermoelasticThermalConductionQuasistatic,
            solid::NodalForcesSolid,
        },
        math::{
            Quantity, Tensor,
            integrate::BackwardEuler,
            optimize::{EqualityConstraint, NewtonRaphson, Tolerances},
        },
        units::{TemperatureRate, Time, VolumetricHeatCapacity},
    };
    const NODES: usize = 5;
    const HEAT_CAPACITY: f64 = 1.0e3;
    const HEATING_RATE: f64 = 0.3;
    const FINAL_TIME: f64 = 1.0;
    const COORDINATES: [[f64; 3]; NODES] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    type Coupled = Canonical<AlmansiHamel, Fourier>;
    type Capacitive = Block<
        Coupled,
        Tetrahedron<4>,
        4,
        3,
        4,
        4,
        crate::fem::block::NoDensity,
        Quantity<VolumetricHeatCapacity>,
    >;
    fn reference() -> NodalReferenceCoordinates<3> {
        COORDINATES.into()
    }
    fn model() -> Model<Capacitive, 3> {
        (
            Block::<Coupled, Tetrahedron<4>, 4, 3, 4, 4>::from((
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
                vec![[0, 1, 2, 3], [1, 2, 3, 4]],
                &reference(),
            ))
            .with_heat_capacity(Quantity::<VolumetricHeatCapacity>::new(HEAT_CAPACITY)),
            reference(),
        )
            .into()
    }
    fn pins() -> Vec<usize> {
        vec![0, 1, 2, 5, 6, 10]
    }
    fn solver(max_steps: usize) -> NewtonRaphson {
        NewtonRaphson {
            abs_tol: Tolerances {
                constraint: 1e-12,
                residual: 1e-6,
            },
            max_steps,
            rel_tol: Some(1e-12),
            ..Default::default()
        }
    }
    fn times(steps: usize) -> Vec<Quantity<Time>> {
        (0..=steps)
            .map(|step| Time::seconds(FINAL_TIME * step as f64 / steps as f64))
            .collect()
    }
    fn heating(model: &Model<Capacitive, 3>) -> NodalForcesThermal {
        model
            .nodal_lumped_heat_capacities()
            .iter()
            .map(|&capacity| capacity * Quantity::<TemperatureRate>::new(HEATING_RATE))
            .collect()
    }
    fn integrate(
        steps: usize,
        constraint: EqualityConstraint,
    ) -> Result<(NodalCoordinates<3>, NodalTemperatures), crate::math::integrate::IntegrationError>
    {
        integrate_with(steps, constraint, solver(25))
    }
    fn integrate_with(
        steps: usize,
        constraint: EqualityConstraint,
        solver: NewtonRaphson,
    ) -> Result<(NodalCoordinates<3>, NodalTemperatures), crate::math::integrate::IntegrationError>
    {
        let model = model();
        let (_, coordinates, _, temperatures, _) =
            ThermoelasticThermalConductionQuasistatic::integrate(
                &model,
                &BackwardEuler::default(),
                solver,
                &times(steps),
                (
                    NodalCoordinates::from(COORDINATES),
                    COORDINATES
                        .map(|_| REFERENCE_TEMPERATURE)
                        .into_iter()
                        .collect(),
                ),
                &NodalForcesSolid::zero(NODES),
                &heating(&model),
                &model.nodal_lumped_heat_capacities(),
                vec![constraint],
            )?;
        Ok((
            coordinates[coordinates.len() - 1].clone(),
            temperatures[temperatures.len() - 1].clone(),
        ))
    }
    fn dilation(temperature: f64) -> f64 {
        let change = temperature - REFERENCE_TEMPERATURE.value();
        (1.0 - 2.0 * COEFFICIENT_OF_THERMAL_EXPANSION.value() * change).powf(-0.5)
    }
    fn biot(temperature: f64) -> f64 {
        9.0 * BULK_MODULUS.value()
            * COEFFICIENT_OF_THERMAL_EXPANSION.value().powi(2)
            * temperature
            * dilation(temperature).powi(2)
    }
    fn backward_euler(steps: usize) -> f64 {
        let dt = FINAL_TIME / steps as f64;
        let alpha = COEFFICIENT_OF_THERMAL_EXPANSION.value();
        let kappa = BULK_MODULUS.value();
        (0..steps).fold(REFERENCE_TEMPERATURE.value(), |previous, _| {
            let residual = |temperature: f64| {
                HEAT_CAPACITY * (temperature - previous) / dt
                    + 9.0
                        * kappa
                        * alpha
                        * temperature
                        * (1.0 - dilation(previous) / dilation(temperature))
                        / dt
                    - HEAT_CAPACITY * HEATING_RATE
            };
            let (mut low, mut high) = (previous, previous + 0.1);
            (0..200).for_each(|_| {
                let middle = 0.5 * (low + high);
                if residual(middle) < 0.0 {
                    low = middle
                } else {
                    high = middle
                }
            });
            0.5 * (low + high)
        })
    }
    fn exact() -> f64 {
        let steps = 20000;
        let dt = FINAL_TIME / steps as f64;
        let rate =
            |temperature: f64| HEAT_CAPACITY * HEATING_RATE / (HEAT_CAPACITY + biot(temperature));
        (0..steps).fold(REFERENCE_TEMPERATURE.value(), |t, _| {
            let k1 = rate(t);
            let k2 = rate(t + 0.5 * dt * k1);
            let k3 = rate(t + 0.5 * dt * k2);
            let k4 = rate(t + dt * k3);
            t + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        })
    }
    #[test]
    fn the_thermoelastic_coupling_slows_the_heating() {
        let uncoupled = HEATING_RATE * FINAL_TIME;
        let change = exact() - REFERENCE_TEMPERATURE.value();
        assert!(change < 0.5 * uncoupled, "{change} !< {uncoupled}");
        assert!(change > 0.0);
    }
    #[test]
    fn the_heating_follows_the_backward_euler_recursion() {
        let steps = 50;
        let (_, temperatures) = integrate(steps, EqualityConstraint::Fixed(pins())).unwrap();
        let expected = backward_euler(steps);
        temperatures.iter().for_each(|temperature| {
            assert!(
                (temperature.value() - expected).abs() < 1e-8,
                "{} != {expected}",
                temperature.value()
            )
        });
    }
    #[test]
    fn the_body_dilates_without_stress_with_the_temperature() {
        let (coordinates, temperatures) = integrate(50, EqualityConstraint::Fixed(pins())).unwrap();
        let dilation = dilation(temperatures[0].value());
        assert!(dilation > 1.0);
        coordinates
            .iter()
            .zip(COORDINATES)
            .for_each(|(coordinate, initial)| {
                (0..3).for_each(|i| {
                    assert!(
                        (coordinate[i].value() - dilation * initial[i]).abs() < 1e-8,
                        "{} != {}",
                        coordinate[i].value(),
                        dilation * initial[i]
                    )
                })
            });
    }
    #[test]
    fn the_integration_approaches_the_exact_heating_at_first_order() {
        let exact = exact() - REFERENCE_TEMPERATURE.value();
        let error = |steps| (backward_euler(steps) - REFERENCE_TEMPERATURE.value() - exact).abs();
        let ratio = error(25) / error(50);
        assert!((ratio - 2.0).abs() < 0.15, "ratio {ratio}");
    }
    #[test]
    fn the_exact_tangent_converges_in_a_few_newton_steps() {
        let (_, temperatures) =
            integrate_with(50, EqualityConstraint::Fixed(pins()), solver(4)).unwrap();
        assert!((temperatures[0].value() - backward_euler(50)).abs() < 1e-8);
    }
    #[test]
    fn a_fixed_temperature_is_held() {
        let (_, temperatures) = integrate(
            10,
            EqualityConstraint::Fixed(pins().into_iter().chain([3]).collect()),
        )
        .unwrap();
        assert_eq!(temperatures[0].value(), REFERENCE_TEMPERATURE.value());
        assert!(temperatures[1].value() > REFERENCE_TEMPERATURE.value());
    }
    #[test]
    fn linear_constraints_are_rejected() {
        assert!(
            integrate(
                10,
                EqualityConstraint::Linear(
                    crate::math::Matrix::zero(1, 4 * NODES),
                    crate::math::Vector::zero(1)
                ),
            )
            .is_err()
        );
    }
}
