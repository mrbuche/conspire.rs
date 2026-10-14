macro_rules! test_thermoelastic_thermal_conduction_time_scale_block {
    ($element: ident) => {
        mod thermoelastic_thermal_conduction_time_scale_block {
            use super::{
                D, G, M, N, P, get_connectivity, get_coordinates_block,
                get_reference_coordinates_block, $element,
            };
            use crate::{
                constitutive::{
                    multiphysics::Canonical,
                    solid::{
                        elastic::AlmansiHamelEulerian,
                        thermoelastic::{
                            AlmansiHamel,
                            test::{
                                BULK_MODULUS, COEFFICIENT_OF_THERMAL_EXPANSION,
                                REFERENCE_TEMPERATURE, SHEAR_MODULUS,
                            },
                        },
                    },
                    thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
                },
                fem::{
                    block::{Block, thermal::NodalTemperatures},
                    multiphysics::solid_thermal::thermoelastic_thermal_conduction::dynamics::time_scale::ThermoelasticThermalConductionTimeScaleElements,
                    thermal::time_scale::ThermalTimeScaleElements,
                    solid::time_scale::TimeScaleElements,
                },
                math::{
                    Quantity,
                    assert::{Assert, AssertionError},
                },
                units::{Density, VolumetricHeatCapacity},
            };
            const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);
            const HEAT_CAPACITY: f64 = 3.0e6;
            fn heat_capacity() -> Quantity<VolumetricHeatCapacity> {
                Quantity::new(HEAT_CAPACITY)
            }
            fn coupled_block() -> Block<
                Canonical<AlmansiHamel, Fourier>,
                $element,
                G,
                M,
                N,
                P,
                Quantity<Density>,
                Quantity<VolumetricHeatCapacity>,
            > {
                Block::<_, _, G, M, N, P, Quantity<Density>>::from((
                    Canonical::from((
                        AlmansiHamel {
                            bulk_modulus: BULK_MODULUS,
                            shear_modulus: SHEAR_MODULUS,
                            coefficient_of_thermal_expansion: COEFFICIENT_OF_THERMAL_EXPANSION,
                            reference_temperature: REFERENCE_TEMPERATURE,
                        },
                        Fourier {
                            thermal_conductivity: THERMAL_CONDUCTIVITY,
                        },
                    )),
                    DENSITY,
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ))
                .with_heat_capacity(heat_capacity())
            }
            fn temperatures() -> NodalTemperatures {
                (0..D).map(|_| REFERENCE_TEMPERATURE).collect()
            }
            #[test]
            fn motion_is_that_of_the_elastic_block_at_the_reference_temperature()
            -> Result<(), AssertionError> {
                let elastic = Block::<AlmansiHamelEulerian, $element, G, M, N, P, Quantity<Density>>::from((
                    AlmansiHamelEulerian {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    DENSITY,
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ));
                let expected = elastic
                    .fastest_time_scale(&get_reference_coordinates_block(), &get_coordinates_block())?;
                let (wave, _) = coupled_block().fastest_time_scales(
                    &get_reference_coordinates_block(),
                    &get_coordinates_block(),
                    &temperatures(),
                )?;
                assert!(expected.value().is_finite());
                Assert::default().eq_within_tols(&wave, &expected)
            }
            #[test]
            fn diffusion_is_that_of_the_thermal_block() -> Result<(), AssertionError> {
                let thermal = Block::<Fourier, $element, G, M, N, P>::from((
                    Fourier {
                        thermal_conductivity: THERMAL_CONDUCTIVITY,
                    },
                    get_connectivity(),
                    &get_reference_coordinates_block(),
                ))
                .with_heat_capacity(heat_capacity());
                let expected = thermal
                    .fastest_diffusive_time_scale(&get_reference_coordinates_block(), &temperatures())?;
                let (_, diffusion) = coupled_block().fastest_time_scales(
                    &get_reference_coordinates_block(),
                    &get_coordinates_block(),
                    &temperatures(),
                )?;
                assert!(expected.value().is_finite());
                Assert::default().eq_within_tols(&diffusion, &expected)
            }
        }
    };
}
pub(crate) use test_thermoelastic_thermal_conduction_time_scale_block;
