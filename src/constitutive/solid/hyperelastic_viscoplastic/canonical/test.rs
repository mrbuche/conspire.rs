use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::viscoplastic::ViscoplasticFlow,
        solid::{
            elastic_viscoplastic::{
                AppliedLoad, ElasticPlasticOrViscoplastic, ElasticViscoplastic,
            },
            hyperelastic::{Hencky, SaintVenantKirchhoff},
        },
    },
    math::{
        Quantity, Tensor, TensorArray,
        assert::{Assert, AssertionError, FiniteDifference, perturbation},
        integrate::{BogackiShampine, DormandPrince, Verner8, Verner9},
        optimize::{GradientDescent, NewtonRaphson},
    },
    mechanics::{CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic},
    units::{Rate, Stress, Time},
};

macro_rules! test_canonical {
    ($elastic:ident) => {
        use super::*;

        fn model() -> Canonical<$elastic, ViscoplasticFlow> {
            Canonical::from((
                $elastic {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                ViscoplasticFlow {
                    yield_stress: Stress::pascals(2.0),
                    hardening_slope: Stress::pascals(1.0),
                    rate_sensitivity: 0.25,
                    reference_flow_rate: Rate::per_second(0.1),
                },
            ))
        }

        #[test]
        fn finite_difference() -> Result<(), AssertionError> {
            let deformation_gradient = DeformationGradient::from([
                [1.31924942, 1.36431217, 0.41764434],
                [0.09959341, 1.38409741, 1.48320137],
                [0.21114106, 1.16675104, 1.98146028],
            ]);
            let deformation_gradient_p = DeformationGradientPlastic::from([
                [0.79610657, 1.36265438, 0.58765375],
                [0.71714877, 1.83110678, 0.69670465],
                [1.82260662, 2.1921719, 3.16928404],
            ]);
            let model = model();
            let tangent =
                model.cauchy_tangent_stiffness(&deformation_gradient, &deformation_gradient_p)?;
            let mut fd = CauchyTangentStiffness::zero();
            for k in 0..3 {
                for l in 0..3 {
                    let mut deformation_gradient_plus = deformation_gradient.clone();
                    deformation_gradient_plus[k][l] += perturbation(0.5 * crate::EPSILON);
                    let cauchy_stress_plus =
                        model.cauchy_stress(&deformation_gradient_plus, &deformation_gradient_p)?;
                    let mut deformation_gradient_minus = deformation_gradient.clone();
                    deformation_gradient_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
                    let cauchy_stress_minus = model
                        .cauchy_stress(&deformation_gradient_minus, &deformation_gradient_p)?;
                    for i in 0..3 {
                        for j in 0..3 {
                            fd[i][j][k][l] = (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j])
                                / crate::EPSILON;
                        }
                    }
                }
            }
            if tangent.error_fd(&fd, 5e1 * crate::EPSILON).is_some() {
                Assert::default().eq_within_fd_tol(&tangent, &fd)
            } else {
                Ok(())
            }
        }

        macro_rules! test_integrator_with_solver {
            ($integrator:ident, $solver:expr) => {
                let model = model();
                let (t, f, f_p) = model.root(
                    AppliedLoad::UniaxialStress(
                        |t: Quantity<Time>| 1.0 + t.value(),
                        &[Quantity::new(0.0), Quantity::new(2.0)],
                    ),
                    $integrator {
                        abs_tol: 1e-6,
                        rel_tol: 1e-6,
                        ..Default::default()
                    },
                    $solver,
                )?;
                for (_, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
                    Assert::non_negative(&model.internal_dissipation(f_i, s_i)?)?;
                }
                let (t, f, f_p) = model.minimize(
                    AppliedLoad::UniaxialStress(
                        |t: Quantity<Time>| 1.0 + t.value(),
                        &[Quantity::new(0.0), Quantity::new(2.0)],
                    ),
                    $integrator {
                        abs_tol: 1e-6,
                        rel_tol: 1e-6,
                        ..Default::default()
                    },
                    $solver,
                )?;
                for (_, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
                    Assert::non_negative(&model.internal_dissipation(f_i, s_i)?)?;
                }
            };
        }

        macro_rules! test_model_with_integrator {
            ($integrator:ident) => {
                #[test]
                fn root_0_and_minimize_1() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::{
                        elastic_viscoplastic::ZerothOrderRoot,
                        hyperelastic_viscoplastic::FirstOrderMinimize,
                    };
                    test_integrator_with_solver!(
                        $integrator,
                        GradientDescent {
                            dual: true,
                            ..Default::default()
                        }
                    );
                    Ok(())
                }
                #[test]
                fn root_1_and_minimize_2() -> Result<(), AssertionError> {
                    use crate::constitutive::solid::{
                        elastic_viscoplastic::FirstOrderRoot,
                        hyperelastic_viscoplastic::SecondOrderMinimize,
                    };
                    test_integrator_with_solver!($integrator, NewtonRaphson::default());
                    Ok(())
                }
            };
        }

        mod bogacki_shampine {
            use super::*;
            test_model_with_integrator!(BogackiShampine);
        }
        mod dormand_prince {
            use super::*;
            test_model_with_integrator!(DormandPrince);
        }
        mod verner_8 {
            use super::*;
            test_model_with_integrator!(Verner8);
        }
        mod verner_9 {
            use super::*;
            test_model_with_integrator!(Verner9);
        }

        mod state_evolution {
            use super::model;
            use crate::{
                constitutive::solid::{
                    elastic_viscoplastic::AppliedLoad,
                    hyperelastic_viscoplastic::{RootRkmkDaeMinimize, SecondOrderMinimize},
                },
                math::{
                    Quantity, Tensor, TensorArray,
                    integrate::{BogackiShampine, BogackiShampineTableau},
                    optimize::NewtonRaphson,
                },
                mechanics::DeformationGradientPlastic,
                units::Time,
            };

            fn time(steps: usize) -> Vec<Quantity<Time>> {
                (0..=steps)
                    .map(|i| Quantity::new(i as f64 / steps as f64))
                    .collect()
            }

            // Resolving F at every stage abscissa by minimization lifts the
            // return map to the tableau's own order, the minimize sibling of
            // `elastic_viscoplastic`'s `rkmk_dae_is_third_order`.
            #[test]
            fn rkmk_dae_minimize_is_third_order() {
                // a modest stretch range -- Saint-Venant-Kirchhoff's tangent
                // degrades away from the asymptotic regime at large stretch
                let load = |t: Quantity<Time>| 1.0 + 0.3 * t.value();
                let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
                let (_, reference, _) = model()
                    .minimize(
                        AppliedLoad::UniaxialStress(load, &span),
                        BogackiShampine {
                            abs_tol: 1e-10,
                            rel_tol: 1e-10,
                            ..Default::default()
                        },
                        NewtonRaphson::default(),
                    )
                    .unwrap();
                let reference = reference.iter().last().unwrap().clone();
                let mut errors = Vec::new();
                // start past the pre-asymptotic regime that a coarser 5-step
                // grid sits in for this stiffer model
                for steps in [10, 20, 40, 80] {
                    let times = time(steps);
                    let (_, dae, state_variables) =
                        RootRkmkDaeMinimize::<Quantity>::root_rkmk_dae_minimize::<
                            BogackiShampineTableau,
                        >(
                            &model(),
                            AppliedLoad::UniaxialStress(load, &times),
                            NewtonRaphson::default(),
                        )
                        .unwrap();
                    let error = (dae.iter().last().unwrap() - &reference).norm().value();
                    println!("{steps}: {error:e}");
                    errors.push(error);
                    let deformation_gradient_p = &state_variables.iter().last().unwrap().0;
                    assert!(
                        (deformation_gradient_p - &DeformationGradientPlastic::identity())
                            .norm()
                            .value()
                            > 1e-3
                    );
                }
                // Bogacki-Shampine is third order, so each halving must cut the error ~8x
                errors.windows(2).for_each(|pair| {
                    let ratio = pair[0] / pair[1];
                    assert!(
                        (6.0..12.0).contains(&ratio),
                        "not third order: {ratio}, {errors:?}"
                    )
                });
            }

            #[test]
            fn rkmk_dae_minimize_adaptive_keeps_the_plastic_deformation_unimodular() {
                let load = |t: Quantity<Time>| 1.0 + 0.3 * t.value();
                let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
                let (times, _, state_variables) =
                    RootRkmkDaeMinimize::<Quantity>::root_rkmk_dae_adaptive_minimize::<
                        BogackiShampineTableau,
                    >(
                        &model(),
                        AppliedLoad::UniaxialStress(load, &span),
                        NewtonRaphson::default(),
                        1e-8,
                        1e-8,
                    )
                    .unwrap();
                assert!(times.len() > 2);
                let deformation_gradient_p = &state_variables.iter().last().unwrap().0;
                assert!((deformation_gradient_p.determinant() - 1.0).abs() < 1e-10);
                assert!(
                    (deformation_gradient_p - &DeformationGradientPlastic::identity())
                        .norm()
                        .value()
                        > 1e-3
                );
            }
        }
    };
}

mod hencky {
    test_canonical!(Hencky);
}
mod saint_venant_kirchhoff {
    test_canonical!(SaintVenantKirchhoff);
}
