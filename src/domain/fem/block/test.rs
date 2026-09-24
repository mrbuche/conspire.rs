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
                            let coordinates = FirstOrderRoot::root(
                                &crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )),
                                EqualityConstraint::Linear(a, b),
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
                        constitutive::solid::elastic::FirstOrderRoot as _, domain::FirstOrderRoot,
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
                            let coordinates = SecondOrderMinimize::minimize(
                                &crate::domain::Model::from((
                                    get_block(),
                                    get_reference_coordinates_block(),
                                )),
                                EqualityConstraint::Linear(a, b),
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
                        constitutive::solid::hyperelastic::SecondOrderMinimize as _,
                        domain::SecondOrderMinimize, math::optimize::NewtonRaphson,
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
                    use crate::constitutive::solid::elastic_hyperviscous::SecondOrderMinimize as _;
                    use crate::fem::solid::elastic_hyperviscous::SecondOrderMinimize;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) =
                        SecondOrderMinimize::minimize(
                            &crate::fem::Model::from((
                                get_block(),
                                get_reference_coordinates_block(),
                            )),
                            EqualityConstraint::Linear(a, b),
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
                    use crate::constitutive::solid::viscoelastic::FirstOrderRoot as _;
                    use crate::fem::solid::viscoelastic::FirstOrderRoot;
                    let (a, b) = applied_velocities();
                    let block = get_block();
                    let (times, coordinates_history, velocities_history) = FirstOrderRoot::root(
                        &crate::fem::Model::from((get_block(), get_reference_coordinates_block())),
                        EqualityConstraint::Linear(a, b),
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
