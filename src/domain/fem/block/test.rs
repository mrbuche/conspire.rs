macro_rules! test_finite_element_block {
    ($element: ident) => {
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type, $element, G, M, N, P> {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                    ))
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
        mod block_plastic {
            use super::*;
            use crate::{
                EPSILON,
                math::{Rank2, TensorRank2, assert::AssertionError},
                mechanics::test::{
                    get_rotation_current_configuration, get_rotation_reference_configuration,
                    get_translation_current_configuration, get_translation_reference_configuration,
                },
            };
            mod elastic_plastic {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical,
 fluid::plastic::{Linear, PlasticFlow, VonMises, Voce},
 solid::{elastic::SaintVenantKirchhoff, hyperelastic::NeoHookean},
                    },
                    domain::block::test::test_finite_element_block_with_elastic_plastic_constitutive_model,
                    fem::solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic_plastic::ElasticPlasticElements},
                };
                type NeoHookeanPlastic = Canonical<NeoHookean, PlasticFlow<VonMises, Linear>>;
                type SaintVenantKirchhoffVocePlastic = Canonical<SaintVenantKirchhoff, PlasticFlow<VonMises, Voce>>;
mod neo_hookean {
                    use super::*;
                    test_finite_element_block_with_elastic_plastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        NeoHookeanPlastic::from((
                            NeoHookean {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            PlasticFlow {
                                surface: VonMises,
                                hardening: Linear {
                                    yield_stress: $crate::units::Stress::pascals(0.01),
                                    hardening_slope: $crate::units::Stress::pascals(1.0),
                                },
                            },
                        )),
                        NeoHookeanPlastic
                    );
                }
mod saint_venant_kirchhoff_voce {
                    use super::*;
                    test_finite_element_block_with_elastic_plastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoffVocePlastic::from((
                            SaintVenantKirchhoff {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            PlasticFlow {
                                surface: VonMises,
                                hardening: Voce {
                                    yield_stress: $crate::units::Stress::pascals(0.01),
                                    hardening_slope: $crate::units::Stress::pascals(1.0),
                                    saturation_stress: $crate::units::Stress::pascals(0.05),
                                    saturation_rate: 50.0,
                                },
                            },
                        )),
                        SaintVenantKirchhoffVocePlastic
                    );
                }

            }
            mod elastic_viscoplastic {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical, fluid::viscoplastic::ViscoplasticFlow,
                        solid::elastic::AlmansiHamelEulerian,
                    },
                    domain::block::test::test_finite_element_block_with_elastic_viscoplastic_constitutive_model,
                    fem::solid::{
                        NodalForcesSolid, NodalStiffnessesSolid,
                        elastic_viscoplastic::ElasticViscoplasticElements,
                    },
                };
                type AlmansiHamel = Canonical<AlmansiHamelEulerian, ViscoplasticFlow>;
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_block_with_elastic_viscoplastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamel::from((
                            AlmansiHamelEulerian {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            ViscoplasticFlow {
                                yield_stress: $crate::units::Stress::pascals(2.0),
                                hardening_slope: $crate::units::Stress::pascals(1.0),
                                rate_sensitivity: 0.25,
                                reference_flow_rate: $crate::units::Rate::per_second(0.1),
                            },
                        )),
                        AlmansiHamel
                    );
                }
            }
            mod hyperelastic_viscoplastic {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical, fluid::viscoplastic::ViscoplasticFlow,
                        solid::hyperelastic::NeoHookean,
                    },
                    domain::block::test::test_finite_element_block_with_hyperelastic_viscoplastic_constitutive_model,
                    fem::solid::{
                        NodalForcesSolid, NodalStiffnessesSolid,
                        elastic_viscoplastic::ElasticViscoplasticElements,
                        hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
                    },
                };
                type NeoHookeanViscoplastic = Canonical<NeoHookean, ViscoplasticFlow>;
                mod neo_hookean {
                    use super::*;
                    test_finite_element_block_with_hyperelastic_viscoplastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        NeoHookeanViscoplastic::from((
                            NeoHookean {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            ViscoplasticFlow {
                                yield_stress: $crate::units::Stress::pascals(2.0),
                                hardening_slope: $crate::units::Stress::pascals(1.0),
                                rate_sensitivity: 0.25,
                                reference_flow_rate: $crate::units::Rate::per_second(0.1),
                            },
                        )),
                        NeoHookeanViscoplastic
                    );
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block;

macro_rules! test_surface_finite_element_block {
    ($element: ident) => {
        use crate::fem::block::element::test::THICKNESS;
        macro_rules! setup_block {
            ($constitutive_model: expr, $constitutive_model_type: ident) => {
                fn get_block() -> Block<$constitutive_model_type, $element, G, M, N, P> {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                        THICKNESS,
                    ))
                }
                fn get_block_transformed() -> Block<$constitutive_model_type, $element, G, M, N, P>
                {
                    Block::<$constitutive_model_type, $element, G, M, N, P>::from((
                        $constitutive_model,
                        get_connectivity(),
                        &get_reference_coordinates_transformed_block(),
                        THICKNESS,
                    ))
                }
            };
        }
        crate::fem::block::test::test_finite_element_block_inner!($element);
    };
}
pub(crate) use test_surface_finite_element_block;

macro_rules! test_finite_element_block_inner {
    ($element: ident) => {
        macro_rules! test_finite_element_block_with_elastic_constitutive_model {
            ($block: ident, $element_: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
                crate::domain::block::test::test_finite_element_block_with_elastic_constitutive_model!(
                    $block,
                    $element_,
                    $constitutive_model,
                    $constitutive_model_type
                );
                macro_rules! test_root_with_solver {
                    ($solver: ident) => {
                        #[test]
                        fn root() -> Result<(), AssertionError> {
                            let (applied_load, a, b) = equality_constraint();
                            let block = get_block();
                            let coordinates = crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )).root(&crate::fem::block::test::displacement_conditions_3(&a, &b, |node, component| get_reference_coordinates_block()[node][component].value()),
                                $solver::default(),
                            )?;
                            let deformation_gradient =
                                $constitutive_model.root(applied_load, $solver::default())?;
                            block
                                .deformation_gradients(&coordinates)
                                .iter()
                                .try_for_each(|deformation_gradients| {
                                    deformation_gradients.iter().try_for_each(
                                        |deformation_gradient_g| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                deformation_gradient_g,
                                                &deformation_gradient,
                                            )
                                        },
                                    )
                                })
                        }
                    };
                }
                mod newton_raphson_root {
                    use super::*;
                    use crate::{
                        constitutive::solid::elastic::Root as _, domain::Root,
                        math::optimize::NewtonRaphson,
                    };
                    test_root_with_solver!(NewtonRaphson);
                }
            };
        }
        macro_rules! test_finite_element_block_with_hyperelastic_constitutive_model {
            ($block: ident, $element_: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
                crate::domain::block::test::test_finite_element_block_with_hyperelastic_constitutive_model!(
                    $block,
                    $element_,
                    $constitutive_model,
                    $constitutive_model_type
                );
                macro_rules! test_minimize_with_solver {
                    ($solver: ident) => {
                        #[test]
                        fn minimize() -> Result<(), AssertionError> {
                            let (applied_load, a, b) = equality_constraint();
                            let block = get_block();
                            let coordinates = crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )).minimize(&crate::fem::block::test::displacement_conditions_3(&a, &b, |node, component| get_reference_coordinates_block()[node][component].value()),
                                $solver::default(),
                            )?;
                            let deformation_gradient =
                                $constitutive_model.minimize(applied_load, $solver::default())?;
                            block
                                .deformation_gradients(&coordinates)
                                .iter()
                                .try_for_each(|deformation_gradients| {
                                    deformation_gradients.iter().try_for_each(
                                        |deformation_gradient_g| {
                                            $crate::math::assert::Assert::default().eq_within_tols(
                                                deformation_gradient_g,
                                                &deformation_gradient,
                                            )
                                        },
                                    )
                                })
                        }
                    };
                }
                mod newton_raphson_minimize {
                    use super::*;
                    use crate::{
                        constitutive::solid::hyperelastic::Minimize as _,
                        domain::Minimize, math::optimize::NewtonRaphson,
                    };
                    test_minimize_with_solver!(NewtonRaphson);
                }
            };
        }
        crate::domain::block::test::test_block_elastic_and_hyperelastic!($element);
        mod block_viscous {
            use super::*;
            use crate::{
                EPSILON,
                fem::block::test::{
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model,
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model,
                },
                math::{Rank2, TensorRank2, assert::AssertionError},
                mechanics::test::{
                    get_rotation_current_configuration, get_rotation_rate_current_configuration,
                    get_rotation_reference_configuration, get_translation_current_configuration,
                    get_translation_rate_current_configuration,
                    get_translation_reference_configuration,
                },
            };
            mod elastic_hyperviscous {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical,
                        fluid::hyperviscous::Newtonian,
                        solid::{
                            elastic::AlmansiHamelEulerian,
                            elastic_hyperviscous::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                        },
                    },
                    fem::{
                        block::solid::SolidElements,
                        solid::{
                            NodalDampingsSolid, elastic_hyperviscous::ElasticHyperviscousElements,
                            viscoelastic::ViscoelasticElements,
                        },
                    },
                };
                type AlmansiHamel = Canonical<AlmansiHamelEulerian, Newtonian>;
                mod almansi_hamel {
                    use super::*;
                    test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
                        ElementBlock,
                        $element,
                        AlmansiHamel::from((
                            AlmansiHamelEulerian {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            Newtonian {
                                bulk_viscosity: BULK_VISCOSITY,
                                shear_viscosity: SHEAR_VISCOSITY,
                            },
                        )),
                        AlmansiHamel
                    );
                }
            }
            mod hyperviscoelastic {
                use super::*;
                use crate::{
                    constitutive::{
                        canonical::Canonical,
                        fluid::hyperviscous::SaintVenantKirchhoff as ViscousSaintVenantKirchhoff,
                        solid::{
                            hyperelastic::SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
                            hyperviscoelastic::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                        },
                    },
                    fem::{
                        block::solid::SolidElements,
                        solid::{
                            NodalDampingsSolid, elastic_hyperviscous::ElasticHyperviscousElements,
                            viscoelastic::ViscoelasticElements,
                        },
                    },
                };
                type SaintVenantKirchhoff =
                    Canonical<HyperelasticSaintVenantKirchhoff, ViscousSaintVenantKirchhoff>;
                mod saint_venant_kirchhoff {
                    use super::*;
                    test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
                        ElementBlock,
                        $element,
                        SaintVenantKirchhoff::from((
                            HyperelasticSaintVenantKirchhoff {
                                bulk_modulus: BULK_MODULUS,
                                shear_modulus: SHEAR_MODULUS,
                            },
                            ViscousSaintVenantKirchhoff {
                                bulk_viscosity: BULK_VISCOSITY,
                                shear_viscosity: SHEAR_VISCOSITY,
                            },
                        )),
                        SaintVenantKirchhoff
                    );
                }
            }
        }
    };
}
pub(crate) use test_finite_element_block_inner;

fn selected(
    matrix: &crate::math::Matrix,
    values: &crate::math::Vector,
    row: usize,
    stride: usize,
) -> (usize, usize, f64) {
    let columns: Vec<usize> = (0..matrix.width())
        .filter(|&column| matrix[row][column] != 0.0)
        .collect();
    assert_eq!(
        columns.len(),
        1,
        "Each row selects a single degree of freedom."
    );
    let column = columns[0];
    assert_eq!(matrix[row][column], 1.0);
    (column / stride, column % stride, values[row])
}

pub(crate) fn velocity_conditions(
    matrix: &crate::math::Matrix,
    values: &crate::math::Vector,
) -> crate::fem::boundary::Rows<crate::fem::boundary::SolidRate<3>> {
    use crate::{
        fem::boundary::{Enforcement, Rows, Velocity},
        units::Velocity as VelocityUnit,
    };
    (0..matrix.len()).fold(Rows::new(Enforcement::Multipliers), |rows, row| {
        let (node, component, value) = selected(matrix, values, row, 3);
        let value = VelocityUnit::meters_per_second(value);
        match component {
            0 => rows.add(Velocity::<0>, node, value),
            1 => rows.add(Velocity::<1>, node, value),
            _ => rows.add(Velocity::<2>, node, value),
        }
    })
}

macro_rules! displacement_conditions {
    ($name:ident, $dimension:literal, [$($axis:literal),*]) => {
        pub(crate) fn $name(
            matrix: &crate::math::Matrix,
            values: &crate::math::Vector,
            reference: impl Fn(usize, usize) -> f64,
        ) -> crate::fem::boundary::Rows<crate::fem::boundary::Solid<$dimension>> {
            use crate::{
                fem::boundary::{Displacement, Enforcement, Rows},
                math::Quantity,
                units::Length,
            };
            (0..matrix.len()).fold(Rows::new(Enforcement::Multipliers), |rows, row| {
                let (node, component, value) = selected(matrix, values, row, $dimension);
                let offset = Quantity::<Length>::new(value - reference(node, component));
                match component {
                    $($axis => rows.add(Displacement::<$axis>, node, offset),)*
                    _ => unreachable!("The component is not one of the dimensions."),
                }
            })
        }
    };
}

displacement_conditions!(displacement_conditions_3, 3, [0, 1, 2]);
displacement_conditions!(displacement_conditions_2, 2, [0, 1]);

pub(crate) fn field_conditions<L, C, I>(
    component: C,
    matrix: &crate::math::Matrix,
    values: &crate::math::Vector,
    value: impl Fn(f64) -> C::Value,
) -> crate::fem::boundary::Rows<L>
where
    L: crate::fem::boundary::Layout,
    C: crate::fem::boundary::Component<L, I> + crate::fem::boundary::Prescribable + Copy,
{
    use crate::fem::boundary::{Enforcement, Rows};
    (0..matrix.len()).fold(Rows::new(Enforcement::Multipliers), |rows, row| {
        let (node, _, entry) = selected(matrix, values, row, 1);
        rows.add(component, node, value(entry))
    })
}

mod conditions {
    use super::{displacement_conditions_3, field_conditions};
    use crate::{
        fem::boundary::{Constrain, Temperature},
        math::{
            Matrix, Quantity, Tensor, Vector,
            optimize::{EqualityConstraint, Rows},
        },
        units::Temperature as TemperatureUnit,
    };
    fn rows() -> (Matrix, Vector) {
        let mut a = Matrix::zero(3, 9);
        a[0][1] = 1.0;
        a[1][5] = 1.0;
        a[2][6] = 1.0;
        (a, Vector::from([0.75, -0.25, 2.0]))
    }
    fn reference(node: usize, component: usize) -> f64 {
        10.0 * node as f64 + component as f64
    }
    fn found(constraints: Vec<EqualityConstraint>, guess: &mut Vector) -> Vec<(usize, f64)> {
        let (eliminated, Rows::Sparse(matrix, values)) =
            EqualityConstraint::split(constraints, guess)
        else {
            panic!("expected sparse rows")
        };
        assert_eq!(eliminated, None);
        let mut found: Vec<(usize, f64)> = (0..values.len())
            .map(|row| {
                let columns: Vec<usize> = (0..9)
                    .filter(|&column| matrix.entry(row, column) != 0.0)
                    .collect();
                assert_eq!(columns.len(), 1);
                (columns[0], values[row])
            })
            .collect();
        found.sort_by_key(|&(column, _)| column);
        found
    }
    #[test]
    fn displacements_come_back_as_the_rows_they_came_from() {
        let (a, b) = rows();
        let mut guess = Vector::zero(9);
        (0..3).for_each(|node| (0..3).for_each(|c| guess[3 * node + c] = reference(node, c)));
        let conditions = displacement_conditions_3(&a, &b, reference);
        assert_eq!(
            found(conditions.constraint(), &mut guess),
            [(1, 0.75), (5, -0.25), (6, 2.0)]
        );
    }
    #[test]
    fn fields_come_back_as_the_rows_they_came_from() {
        let (a, b) = rows();
        let mut guess = Vector::zero(9);
        let conditions = field_conditions(Temperature, &a, &b, Quantity::<TemperatureUnit>::new);
        let held: crate::fem::boundary::Rows<crate::fem::boundary::Field<TemperatureUnit>> =
            conditions;
        assert_eq!(
            found(held.constraint(), &mut guess),
            [(1, 0.75), (5, -0.25), (6, 2.0)]
        );
    }
}

macro_rules! test_dae_root_and_minimize_with_integrators {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        use crate::math::{
            Quantity,
            integrate::{BogackiShampine, DormandPrince, Verner8, Verner9},
            optimize::NewtonRaphson,
        };
        macro_rules! test_with_integrator {
            ($integrator: ident) => {
                #[test]
                fn minimize() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::elastic_hyperviscous::Minimize as _;
                    use crate::fem::solid::elastic_hyperviscous::Minimize;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) =
                        crate::fem::Model::from((get_block(), get_reference_coordinates_block()))
                            .minimize(
                            &crate::fem::block::test::velocity_conditions(&a, &b),
                            $integrator::default(),
                            &[Quantity::new(0.0), Quantity::new(1.0)],
                            NewtonRaphson::default(),
                        )?;
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model.minimize(
                            applied_velocity(&times),
                            $integrator::default(),
                            NewtonRaphson::default(),
                        )?;
                    coordinates_history
                        .iter()
                        .zip(
                            velocities_history.iter().zip(
                                deformation_gradients
                                    .iter()
                                    .zip(deformation_gradient_rates.iter()),
                            ),
                        )
                        .try_for_each(
                            |(
                                coordinates,
                                (velocities, (deformation_gradient, deformation_gradient_rate)),
                            )| {
                                block
                                    .deformation_gradients(coordinates)
                                    .iter()
                                    .try_for_each(|deformation_gradients| {
                                        deformation_gradients.iter().try_for_each(
                                            |deformation_gradient_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_g,
                                                        deformation_gradient,
                                                    )
                                            },
                                        )
                                    })?;
                                block
                                    .deformation_gradient_rates(coordinates, velocities)
                                    .iter()
                                    .try_for_each(|deformation_gradient_rates| {
                                        deformation_gradient_rates.iter().try_for_each(
                                            |deformation_gradient_rate_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_rate_g,
                                                        deformation_gradient_rate,
                                                    )
                                            },
                                        )
                                    })
                            },
                        )
                }
                #[test]
                fn root() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::viscoelastic::Root as _;
                    use crate::fem::solid::viscoelastic::Root;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) =
                        crate::fem::Model::from((get_block(), get_reference_coordinates_block()))
                            .root(
                            &crate::fem::block::test::velocity_conditions(&a, &b),
                            $integrator::default(),
                            &[Quantity::new(0.0), Quantity::new(1.0)],
                            NewtonRaphson::default(),
                        )?;
                    let (_, deformation_gradients, deformation_gradient_rates) =
                        $constitutive_model.root(
                            applied_velocity(&times),
                            $integrator::default(),
                            NewtonRaphson::default(),
                        )?;
                    coordinates_history
                        .iter()
                        .zip(
                            velocities_history.iter().zip(
                                deformation_gradients
                                    .iter()
                                    .zip(deformation_gradient_rates.iter()),
                            ),
                        )
                        .try_for_each(
                            |(
                                coordinates,
                                (velocities, (deformation_gradient, deformation_gradient_rate)),
                            )| {
                                block
                                    .deformation_gradients(coordinates)
                                    .iter()
                                    .try_for_each(|deformation_gradients| {
                                        deformation_gradients.iter().try_for_each(
                                            |deformation_gradient_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_g,
                                                        deformation_gradient,
                                                    )
                                            },
                                        )
                                    })?;
                                block
                                    .deformation_gradient_rates(coordinates, velocities)
                                    .iter()
                                    .try_for_each(|deformation_gradient_rates| {
                                        deformation_gradient_rates.iter().try_for_each(
                                            |deformation_gradient_rate_g| {
                                                $crate::math::assert::Assert::default()
                                                    .eq_within_tols(
                                                        deformation_gradient_rate_g,
                                                        deformation_gradient_rate,
                                                    )
                                            },
                                        )
                                    })
                            },
                        )
                }
            };
        }
        mod bogacki_shampine {
            use super::*;
            test_with_integrator!(BogackiShampine);
        }
        mod dormand_prince {
            use super::*;
            test_with_integrator!(DormandPrince);
        }
        mod verner_8 {
            use super::*;
            test_with_integrator!(Verner8);
        }
        mod verner_9 {
            use super::*;
            test_with_integrator!(Verner9);
        }
    };
}
pub(crate) use test_dae_root_and_minimize_with_integrators;

macro_rules! test_finite_element_block_with_elastic_hyperviscous_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::domain::block::test::test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        crate::fem::block::test::test_dae_root_and_minimize_with_integrators!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_block_with_elastic_hyperviscous_constitutive_model;

macro_rules! test_finite_element_block_with_hyperviscoelastic_constitutive_model {
    ($block: ident, $element: ident, $constitutive_model: expr, $constitutive_model_type: ident) => {
        crate::domain::block::test::test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
        crate::fem::block::test::test_dae_root_and_minimize_with_integrators!(
            $block,
            $element,
            $constitutive_model,
            $constitutive_model_type
        );
    };
}
pub(crate) use test_finite_element_block_with_hyperviscoelastic_constitutive_model;
