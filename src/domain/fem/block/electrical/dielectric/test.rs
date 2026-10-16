macro_rules! test_dielectric_block {
    ($element: ident) => {
        mod dielectric_block {
            use super::{
                D, G, M, N, P, get_connectivity, get_reference_coordinates_block, $element,
            };
            use crate::{
                EPSILON,
                constitutive::electrical::dielectric::Linear,
                domain::diffusion::DiffusionElements,
                fem::{
                    ElementModelError,
                    block::{
                        Block,
                        electrical::{
                            NodalPotentials,
                            dielectric::{NodalForcesElectrical, NodalStiffnessesElectrical},
                        },
                    },
                },
                math::{Quantity, assert::AssertionError},
            };
            mod finite_difference {
                use super::*;
                use crate::units::{ElectricPotential, Permittivity};
                const MODEL: Linear = Linear {
                    permittivity: Permittivity::farads_per_meter(2.5),
                };
                const EPSILON_POTENTIAL: Quantity<ElectricPotential> =
                    ElectricPotential::volts(EPSILON);
                fn potentials() -> NodalPotentials {
                    (0..D)
                        .map(|node| ElectricPotential::volts((1.0 + node as f64).sin()))
                        .collect()
                }
                fn block() -> Block<Linear, $element, G, M, N, P> {
                    Block::<Linear, $element, G, M, N, P>::from((
                        MODEL,
                        get_connectivity(),
                        &get_reference_coordinates_block(),
                    ))
                }
                #[test]
                fn potential() -> Result<(), AssertionError> {
                    let block = block();
                    let mut finite_difference = Quantity::default();
                    let nodal_forces_fd: NodalForcesElectrical = (0..D)
                        .map(|node| {
                            let mut nodal_potentials = potentials();
                            nodal_potentials[node] += EPSILON_POTENTIAL * 0.5;
                            finite_difference = block.potential(&nodal_potentials)?;
                            nodal_potentials[node] -= EPSILON_POTENTIAL;
                            finite_difference -= block.potential(&nodal_potentials)?;
                            Ok(finite_difference / EPSILON_POTENTIAL)
                        })
                        .collect::<Result<_, ElementModelError>>()?;
                    $crate::math::assert::Assert::default()
                        .eq_within_fd_tol(&nodal_forces_fd, &block.nodal_forces(&potentials())?)
                }
                #[test]
                fn nodal_forces() -> Result<(), AssertionError> {
                    let block = block();
                    let mut finite_difference = Quantity::default();
                    let nodal_stiffnesses_fd: NodalStiffnessesElectrical = (0..D)
                        .map(|node_a| {
                            (0..D)
                                .map(|node_b| {
                                    let mut nodal_potentials = potentials();
                                    nodal_potentials[node_b] += EPSILON_POTENTIAL * 0.5;
                                    finite_difference =
                                        block.nodal_forces(&nodal_potentials)?[node_a];
                                    nodal_potentials[node_b] -= EPSILON_POTENTIAL;
                                    finite_difference -=
                                        block.nodal_forces(&nodal_potentials)?[node_a];
                                    Ok(finite_difference / EPSILON_POTENTIAL)
                                })
                                .collect()
                        })
                        .collect::<Result<_, ElementModelError>>()?;
                    $crate::math::assert::Assert::default().eq_within_fd_tol(
                        &nodal_stiffnesses_fd,
                        &block.nodal_stiffnesses(&potentials())?,
                    )
                }
            }
        }
    };
}
pub(crate) use test_dielectric_block;
