use crate::{
    constitutive::{
        canonical::Canonical,
        fluid::viscoplastic::ViscoplasticFlow,
        solid::{
            elastic::AlmansiHamelEulerian,
            elastic_viscoplastic::{
                AppliedLoad, ElasticPlasticOrViscoplastic, ElasticViscoplastic,
            },
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

fn model() -> Canonical<AlmansiHamelEulerian, ViscoplasticFlow> {
    Canonical::from((
        AlmansiHamelEulerian {
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
    let tangent = model.cauchy_tangent_stiffness(&deformation_gradient, &deformation_gradient_p)?;
    let mut fd = CauchyTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let cauchy_stress_plus =
                model.cauchy_stress(&deformation_gradient_plus, &deformation_gradient_p)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let cauchy_stress_minus =
                model.cauchy_stress(&deformation_gradient_minus, &deformation_gradient_p)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] =
                        (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j]) / crate::EPSILON;
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

fn deformation_gradients() -> (DeformationGradient, DeformationGradientPlastic) {
    (
        DeformationGradient::from([
            [1.31924942, 1.36431217, 0.41764434],
            [0.09959341, 1.38409741, 1.48320137],
            [0.21114106, 1.16675104, 1.98146028],
        ]),
        DeformationGradientPlastic::from([
            [0.79610657, 1.36265438, 0.58765375],
            [0.71714877, 1.83110678, 0.69670465],
            [1.82260662, 2.1921719, 3.16928404],
        ]),
    )
}

#[test]
fn mandel_stress_tangent_matches_finite_difference() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_viscoplastic::PlasticTangents,
        mechanics::MandelStressTangentElastic,
    };
    let (deformation_gradient, deformation_gradient_p) = deformation_gradients();
    let model = model();
    let tangent = model.mandel_stress_tangent(&deformation_gradient, &deformation_gradient_p)?;
    let mut fd = MandelStressTangentElastic::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut plus = deformation_gradient.clone();
            plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let mandel_plus = model.mandel_stress(&plus, &deformation_gradient_p)?;
            let mut minus = deformation_gradient.clone();
            minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let mandel_minus = model.mandel_stress(&minus, &deformation_gradient_p)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (mandel_plus[i][j] - mandel_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    Assert::default().eq_within_fd_tol(&tangent, &fd)
}

#[test]
fn mandel_stress_tangent_p_matches_finite_difference() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_viscoplastic::PlasticTangents,
        mechanics::MandelStressTangentElasticPlastic,
    };
    let (deformation_gradient, deformation_gradient_p) = deformation_gradients();
    let model = model();
    let tangent = model.mandel_stress_tangent_p(&deformation_gradient, &deformation_gradient_p)?;
    let mut fd = MandelStressTangentElasticPlastic::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut plus = deformation_gradient_p.clone();
            plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let mandel_plus = model.mandel_stress(&deformation_gradient, &plus)?;
            let mut minus = deformation_gradient_p.clone();
            minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let mandel_minus = model.mandel_stress(&deformation_gradient, &minus)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (mandel_plus[i][j] - mandel_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    Assert::default().eq_within_fd_tol(&tangent, &fd)
}

#[test]
fn cauchy_tangent_stiffness_p_matches_finite_difference() -> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_viscoplastic::PlasticTangents,
        mechanics::CauchyTangentStiffnessPlastic,
    };
    let (deformation_gradient, deformation_gradient_p) = deformation_gradients();
    let model = model();
    let tangent =
        model.cauchy_tangent_stiffness_p(&deformation_gradient, &deformation_gradient_p)?;
    let mut fd = CauchyTangentStiffnessPlastic::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut plus = deformation_gradient_p.clone();
            plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let stress_plus = model.cauchy_stress(&deformation_gradient, &plus)?;
            let mut minus = deformation_gradient_p.clone();
            minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let stress_minus = model.cauchy_stress(&deformation_gradient, &minus)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (stress_plus[i][j] - stress_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    Assert::default().eq_within_fd_tol(&tangent, &fd)
}

#[test]
fn first_piola_kirchhoff_tangent_stiffness_p_matches_finite_difference()
-> Result<(), AssertionError> {
    use crate::{
        constitutive::solid::elastic_viscoplastic::PlasticTangents,
        mechanics::FirstPiolaKirchhoffTangentStiffnessPlastic,
    };
    let (deformation_gradient, deformation_gradient_p) = deformation_gradients();
    let model = model();
    let tangent = model.first_piola_kirchhoff_tangent_stiffness_p(
        &deformation_gradient,
        &deformation_gradient_p,
    )?;
    let mut fd = FirstPiolaKirchhoffTangentStiffnessPlastic::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut plus = deformation_gradient_p.clone();
            plus[k][l] += perturbation(0.5 * crate::EPSILON);
            let stress_plus = model.first_piola_kirchhoff_stress(&deformation_gradient, &plus)?;
            let mut minus = deformation_gradient_p.clone();
            minus[k][l] -= perturbation(0.5 * crate::EPSILON);
            let stress_minus = model.first_piola_kirchhoff_stress(&deformation_gradient, &minus)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] = (stress_plus[i][j] - stress_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    Assert::default().eq_within_fd_tol(&tangent, &fd)
}

macro_rules! test_integrator_with_solver {
    ($integrator:ident, $solver:expr, $final_time:literal) => {
        let model = model();
        let (t, f, f_p) = model.root(
            AppliedLoad::UniaxialStress(
                |t: Quantity<Time>| 1.0 + t.value(),
                &[Quantity::new(0.0), Quantity::new($final_time)],
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
        fn root_0() -> Result<(), AssertionError> {
            use crate::constitutive::solid::elastic_viscoplastic::ZerothOrderRoot;
            test_integrator_with_solver!(
                $integrator,
                GradientDescent {
                    dual: true,
                    ..Default::default()
                },
                0.5
            );
            Ok(())
        }
        #[test]
        fn root_1() -> Result<(), AssertionError> {
            use crate::constitutive::solid::elastic_viscoplastic::FirstOrderRoot;
            test_integrator_with_solver!($integrator, NewtonRaphson::default(), 2.0);
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
        constitutive::solid::elastic_viscoplastic::ViscoplasticStateVariables,
        math::{
            Quantity, Tensor, TensorArray, TensorTuple, TensorVector,
            assert::{Assert, AssertionError},
            integrate::{
                BogackiShampineTableau, StateEvolution, Times, integrate_rkmk_state,
                integrate_rkmk_state_adaptive,
            },
        },
        mechanics::{DeformationGradient, DeformationGradientPlastic},
        units::Time,
    };

    // simple shear, det = 1; large enough that the deviatoric Mandel stress yields
    fn deformation_gradient() -> DeformationGradient {
        DeformationGradient::from([[1.0, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    fn time(steps: usize) -> Vec<Quantity<Time>> {
        (0..=steps)
            .map(|i| Quantity::new(i as f64 / steps as f64))
            .collect()
    }

    #[test]
    fn rkmk_state_keeps_the_plastic_deformation_unimodular() {
        let model = model();
        let (_, states): (Times, TensorVector<_>) =
            integrate_rkmk_state::<_, BogackiShampineTableau, _, _, _>(
                &model,
                |_| deformation_gradient(),
                &time(20),
            )
            .unwrap();
        let final_state = states.iter().last().unwrap();
        assert!((final_state.0.determinant() - 1.0).abs() < 1e-10);
        // and F_p actually flowed
        assert!(
            (&final_state.0 - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
    }

    #[test]
    fn rkmk_state_adaptive_keeps_the_plastic_deformation_unimodular_and_meets_tolerance() {
        let model = model();
        let (times, states): (Times, TensorVector<_>) =
            integrate_rkmk_state_adaptive::<_, BogackiShampineTableau, _, _, _>(
                &model,
                |_| deformation_gradient(),
                &time(1),
                1e-8,
                1e-8,
            )
            .unwrap();
        // the controller subdivided the single [0, 1] span
        assert!(times.len() > 2);
        let final_state = states.iter().last().unwrap();
        assert!((final_state.0.determinant() - 1.0).abs() < 1e-10);
        assert!(
            (&final_state.0 - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
        // a much looser tolerance takes fewer steps
        let (loose_times, _): (Times, TensorVector<_>) =
            integrate_rkmk_state_adaptive::<_, BogackiShampineTableau, _, _, _>(
                &model,
                |_| deformation_gradient(),
                &time(1),
                1e-3,
                1e-3,
            )
            .unwrap();
        assert!(loose_times.len() < times.len());
    }

    #[test]
    fn the_legacy_additive_march_drifts_off_the_group_where_rkmk_does_not() {
        let model = model();
        let f = deformation_gradient();
        let steps = time(20);
        let mut fp = DeformationGradientPlastic::identity();
        let mut eps = Quantity::new(0.0);
        for w in steps.windows(2) {
            let dt = w[1] - w[0];
            let rate = StateEvolution::state_rate(&model, w[0], &f, &TensorTuple(fp.clone(), eps))
                .unwrap();
            // forward Euler on Ḟ_p = D_p F_p — never re-projected onto the group
            fp = &(&rate.0 * &fp) * dt + &fp;
            eps += rate.1 * dt;
        }
        assert!((fp.determinant() - 1.0).abs() > 1e-4);
    }

    #[test]
    fn root_rkmk_solves_uniaxial_stress_and_keeps_f_p_unimodular() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, RkmkRoot},
            math::optimize::NewtonRaphson,
        };
        let times = time(16);
        let (_, _, state_variables) = model()
            .root_rkmk::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(|t: Quantity<Time>| 1.0 + 4.0 * t.value(), &times),
                NewtonRaphson::default(),
            )
            .unwrap();
        let deformation_gradient_p = &state_variables.iter().last().unwrap().0;
        // the operator-split RKMK step keeps det F_p = 1 through the coupled solve
        assert!((deformation_gradient_p.determinant() - 1.0).abs() < 1e-10);
        // and the plastic state actually flowed under the applied load
        assert!(
            (deformation_gradient_p - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
    }

    // The adaptive group leg substeps within each load window but keeps the same
    // operator-split coupling as the fixed step, so on a moderately fine load grid
    // (where one BS step per window is already accurate) the two must agree.
    #[test]
    fn root_rkmk_adaptive_agrees_with_the_fixed_step_on_a_fine_grid() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, RkmkRoot},
            math::optimize::NewtonRaphson,
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let times = time(24);
        let (_, deformation_gradients_adaptive, state_variables_adaptive) = model()
            .root_rkmk_adaptive::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(load, &times),
                NewtonRaphson::default(),
                1e-9,
                1e-9,
            )
            .unwrap();
        let (_, deformation_gradients_fixed, state_variables_fixed) = model()
            .root_rkmk::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(load, &times),
                NewtonRaphson::default(),
            )
            .unwrap();
        let f_p_adaptive = &state_variables_adaptive.iter().last().unwrap().0;
        let f_p_fixed = &state_variables_fixed.iter().last().unwrap().0;
        assert!((f_p_adaptive.determinant() - 1.0).abs() < 1e-10);
        assert!((f_p_adaptive - f_p_fixed).norm().value() < 5e-4);
        assert!(
            (deformation_gradients_adaptive.iter().last().unwrap()
                - deformation_gradients_fixed.iter().last().unwrap())
            .norm()
            .value()
                < 5e-4
        );
        assert!(
            (f_p_adaptive - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
    }

    // The operator split is only first order in the F <-> F_p coupling, so as the
    // step shrinks its solution must approach the monolithic DAE first-order root,
    // and its own step-to-step change must halve as dt halves.
    #[test]
    fn root_rkmk_converges_to_the_first_order_root_under_step_refinement() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, FirstOrderRoot, RkmkRoot},
            math::{integrate::BogackiShampine, optimize::NewtonRaphson},
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
        let (_, deformation_gradients_reference, _) = model()
            .root(
                AppliedLoad::UniaxialStress(load, &span),
                BogackiShampine {
                    abs_tol: 1e-8,
                    rel_tol: 1e-8,
                    ..Default::default()
                },
                NewtonRaphson::default(),
            )
            .unwrap();
        let reference = deformation_gradients_reference
            .iter()
            .last()
            .unwrap()
            .clone();
        let final_deformation_gradient = |steps: usize| {
            let times: Vec<Quantity<Time>> = (0..=steps)
                .map(|i| Quantity::new(i as f64 / steps as f64))
                .collect();
            let (_, deformation_gradients, _) = model()
                .root_rkmk::<BogackiShampineTableau>(
                    AppliedLoad::UniaxialStress(load, &times),
                    NewtonRaphson::default(),
                )
                .unwrap();
            deformation_gradients.iter().last().unwrap().clone()
        };
        let f_10 = final_deformation_gradient(10);
        let f_20 = final_deformation_gradient(20);
        let f_40 = final_deformation_gradient(40);
        let f_80 = final_deformation_gradient(80);
        let e_20 = (&f_20 - &reference).norm().value();
        let e_40 = (&f_40 - &reference).norm().value();
        let e_80 = (&f_80 - &reference).norm().value();
        assert!(
            e_40 < e_20 && e_80 < e_40,
            "not monotone toward the DAE root: {e_20}, {e_40}, {e_80}"
        );
        assert!(e_80 < e_20 / 2.0, "not converging: {e_20} -> {e_80}");
        assert!(e_80 < 1e-2, "n=80 disagreement with the DAE root: {e_80}");
        let ratio_1 = (&f_20 - &f_10).norm().value() / (&f_40 - &f_20).norm().value();
        let ratio_2 = (&f_40 - &f_20).norm().value() / (&f_80 - &f_40).norm().value();
        assert!(
            (1.4..2.8).contains(&ratio_1) && (1.4..2.8).contains(&ratio_2),
            "self-convergence not first order: {ratio_1}, {ratio_2}"
        );
    }

    // The recorded (F, state) pairs must not violate the second law along the
    // whole loading history.
    #[test]
    fn root_rkmk_keeps_the_internal_dissipation_non_negative() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{
                AppliedLoad, ElasticViscoplastic, RkmkRoot,
            },
            math::{assert::Assert, optimize::NewtonRaphson},
        };
        let times = time(24);
        let model = model();
        let (_, deformation_gradients, state_variables) = model
            .root_rkmk::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(|t: Quantity<Time>| 1.0 + 2.0 * t.value(), &times),
                NewtonRaphson::default(),
            )
            .unwrap();
        deformation_gradients
            .iter()
            .zip(state_variables.iter())
            .for_each(|(deformation_gradient, state)| {
                Assert::non_negative(
                    &model
                        .internal_dissipation(deformation_gradient, state)
                        .unwrap(),
                )
                .unwrap()
            });
    }

    // Cost of one material-point RKMK step vs its four rate evaluations alone
    // (Bogacki–Shampine has four stages) — the difference is the `expm`/`dexpinv`
    // overhead. Prints the ratio; asserts only a loose gross-regression bound
    // since wall-clock timing is noisy.
    #[test]
    fn rkmk_step_cost_relative_to_the_rate_evaluations_alone() {
        use crate::math::integrate::{StateEvolution, rkmk_step};
        use std::time::Instant;
        type Model = super::Canonical<super::AlmansiHamelEulerian, super::ViscoplasticFlow>;
        type Field = <Model as StateEvolution<Time>>::Field;
        let model = model();
        let f = deformation_gradient();
        let t = Quantity::<Time>::new(0.0);
        let dt = Quantity::<Time>::new(0.05);
        let initial = <Model as StateEvolution<Time>>::initial_state(&model);
        let iterations = 20_000;
        let mut scratch = Vec::new();
        let mut sink = 0.0;
        // warm-up
        for _ in 0..2_000 {
            sink += rkmk_step::<Field, BogackiShampineTableau, Time>(
                &mut |tt, s| model.state_rate(tt, &f, s),
                &initial,
                t,
                dt,
                &mut scratch,
            )
            .unwrap()
            .0
            .determinant();
        }
        let start = Instant::now();
        for _ in 0..iterations {
            sink += rkmk_step::<Field, BogackiShampineTableau, Time>(
                &mut |tt, s| model.state_rate(tt, &f, s),
                &initial,
                t,
                dt,
                &mut scratch,
            )
            .unwrap()
            .0
            .determinant();
        }
        let rkmk = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..4 {
                let rate = StateEvolution::state_rate(&model, t, &f, &initial).unwrap();
                sink += rate.0.norm().value();
            }
        }
        let rates = start.elapsed();
        println!(
            "rkmk_step {rkmk:?}  vs  4x state_rate {rates:?}  =>  {:.2}x",
            rkmk.as_secs_f64() / rates.as_secs_f64()
        );
        assert!(sink.is_finite());
        assert!(rkmk.as_secs_f64() < 20.0 * rates.as_secs_f64());
    }

    type Model = super::Canonical<super::AlmansiHamelEulerian, super::ViscoplasticFlow>;
    type Field = <Model as StateEvolution<Time>>::Field;

    fn advance(
        model: &Model,
        deformation_gradient: &DeformationGradient,
        state: &ViscoplasticStateVariables<Quantity>,
        time_step: Quantity<Time>,
    ) -> ViscoplasticStateVariables<Quantity> {
        crate::math::integrate::rkmk_step::<Field, BogackiShampineTableau, Time>(
            &mut |t, point| model.state_rate(t, deformation_gradient, point),
            state,
            Quantity::new(0.0),
            time_step,
            &mut Vec::new(),
        )
        .unwrap()
    }

    // an evolved state, so F_p is off the identity and the hardening has accrued
    fn evolved_state(
        model: &Model,
        deformation_gradient: &DeformationGradient,
        time_step: Quantity<Time>,
    ) -> ViscoplasticStateVariables<Quantity> {
        let mut state = <Model as StateEvolution<Time>>::initial_state(model);
        for _ in 0..3 {
            state = advance(model, deformation_gradient, &state, time_step)
        }
        state
    }

    #[test]
    fn rkmk_step_tangent_matches_finite_difference() -> Result<(), AssertionError> {
        use crate::math::{
            Current, Intermediate, Reference, TensorRank2, TensorRank4, assert::perturbation,
        };
        let model = model();
        let deformation_gradient = deformation_gradient();
        let time_step = Quantity::<Time>::new(0.25);
        let state = evolved_state(&model, &deformation_gradient, time_step);
        let tangent = model
            .rkmk_step_tangent::<BogackiShampineTableau>(&deformation_gradient, &state, time_step)
            .unwrap();
        let reference = advance(&model, &deformation_gradient, &state, time_step);
        Assert::default().eq_within_tols(&tangent.state.0, &reference.0)?;
        Assert::default().eq_within_tols(tangent.state.1, &reference.1)?;
        let mut plastic_difference =
            TensorRank4::<3, Intermediate, Reference, Current, Reference>::zero();
        let mut hardening_difference = TensorRank2::<3, Current, Reference>::zero();
        for k in 0..3 {
            for l in 0..3 {
                let mut plus = deformation_gradient.clone();
                plus[k][l] += perturbation(0.5 * crate::EPSILON);
                let plus = advance(&model, &plus, &state, time_step);
                let mut minus = deformation_gradient.clone();
                minus[k][l] -= perturbation(0.5 * crate::EPSILON);
                let minus = advance(&model, &minus, &state, time_step);
                for a in 0..3 {
                    for b in 0..3 {
                        plastic_difference[a][b][k][l] =
                            (plus.0[a][b] - minus.0[a][b]) / crate::EPSILON
                    }
                }
                hardening_difference[k][l] = (plus.1 - minus.1) / crate::EPSILON
            }
        }
        // not vacuous: the step really does move with F
        assert!(plastic_difference.norm().value() > 1e-3);
        assert!(hardening_difference.norm().value() > 1e-3);
        Assert::default()
            .eq_within_fd_tol(&tangent.deformation_gradient_p_tangent, &plastic_difference)?;
        Assert::default().eq_within_fd_tol(&tangent.hardening_tangent, &hardening_difference)
    }

    #[test]
    fn coupled_tangent_matches_finite_difference() -> Result<(), AssertionError> {
        use crate::{math::assert::perturbation, mechanics::FirstPiolaKirchhoffTangentStiffness};
        let model = model();
        let deformation_gradient = deformation_gradient();
        let time_step = Quantity::<Time>::new(0.25);
        let state = evolved_state(&model, &deformation_gradient, time_step);
        let stress_and_tangent = |deformation_gradient: &DeformationGradient| {
            model
                .first_piola_kirchhoff_stress_rkmk::<BogackiShampineTableau>(
                    deformation_gradient,
                    &state,
                    time_step,
                )
                .unwrap()
        };
        let tangent = stress_and_tangent(&deformation_gradient).1;
        let mut difference = FirstPiolaKirchhoffTangentStiffness::zero();
        for k in 0..3 {
            for l in 0..3 {
                let mut plus = deformation_gradient.clone();
                plus[k][l] += perturbation(0.5 * crate::EPSILON);
                let plus = stress_and_tangent(&plus).0;
                let mut minus = deformation_gradient.clone();
                minus[k][l] -= perturbation(0.5 * crate::EPSILON);
                let minus = stress_and_tangent(&minus).0;
                for i in 0..3 {
                    for j in 0..3 {
                        difference[i][j][k][l] = (plus[i][j] - minus[i][j]) / crate::EPSILON
                    }
                }
            }
        }
        Assert::default().eq_within_fd_tol(&tangent, &difference)
    }

    // The algorithmic tangent is exact, so a Newton on the coupled residual
    // P(F, F_p^{n+1}(F)) = P* must converge quadratically.
    #[test]
    fn coupled_newton_converges_quadratically() {
        use crate::math::{SquareMatrix, Vector};
        let model = model();
        let time_step = Quantity::<Time>::new(0.25);
        let state = evolved_state(&model, &deformation_gradient(), time_step);
        let target = DeformationGradient::from([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let stress = model
            .first_piola_kirchhoff_stress_rkmk::<BogackiShampineTableau>(&target, &state, time_step)
            .unwrap()
            .0;
        let scale = stress.norm().value();
        let mut deformation_gradient = DeformationGradient::from([
            [1.002, 0.39, 0.003],
            [0.0, 0.998, -0.003],
            [0.0, 0.003, 1.0],
        ]);
        let mut errors = Vec::new();
        for _ in 0..6 {
            let (predicted, tangent, _) = model
                .first_piola_kirchhoff_stress_rkmk::<BogackiShampineTableau>(
                    &deformation_gradient,
                    &state,
                    time_step,
                )
                .unwrap();
            let residual = predicted - &stress;
            errors.push(residual.norm().value() / scale);
            if errors.last().unwrap() < &1e-14 {
                break;
            }
            let mut matrix = SquareMatrix::zero(9);
            let mut right_hand_side = Vector::zero(9);
            for i in 0..3 {
                for j in 0..3 {
                    right_hand_side[3 * i + j] = -residual[i][j].value();
                    for k in 0..3 {
                        for l in 0..3 {
                            matrix[3 * i + j][3 * k + l] = tangent[i][j][k][l].value()
                        }
                    }
                }
            }
            let increment = matrix.solve_lu(&right_hand_side).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    deformation_gradient[i][j] += Quantity::new(increment[3 * i + j])
                }
            }
        }
        println!("relative residuals: {errors:?}");
        assert!(
            errors.windows(2).all(|pair| pair[1] < pair[0]),
            "not monotone: {errors:?}"
        );
        assert!(*errors.last().unwrap() < 1e-12, "not converged: {errors:?}");
        // e_{k+1} <= C e_k^2 across the whole asymptotic tail, above round-off
        let asymptotic: Vec<usize> = (0..errors.len() - 1)
            .filter(|k| errors[*k] < 1e-3 && errors[k + 1] > 1e-13)
            .collect();
        assert!(asymptotic.len() > 1, "too short a tail: {errors:?}");
        asymptotic.iter().for_each(|k| {
            assert!(
                errors[k + 1] < 50.0 * errors[*k].powi(2),
                "not quadratic at {k}: {errors:?}"
            )
        });
    }

    // Both maps freeze F across a window, so both stay first order in dt; what
    // separates them is which end they freeze at. The split drives the flow with
    // the F converged at the *start* of the window, the coupled map with the F
    // that equilibrates at its *end*, so their endpoints must straddle the
    // refined reference — the sign check is what proves the coupling is real and
    // not a relabelled split.
    #[test]
    fn coupled_straddles_the_refined_reference_opposite_the_operator_split() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, RkmkRoot},
            math::optimize::NewtonRaphson,
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let reference_times = time(2000);
        let (_, reference, _) = model()
            .root_rkmk::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(load, &reference_times),
                NewtonRaphson::default(),
            )
            .unwrap();
        let reference = reference.iter().last().unwrap().clone();
        let mut errors = Vec::new();
        for steps in [10, 20, 40, 80] {
            let times = time(steps);
            let (_, coupled, state_variables) = model()
                .root_rkmk_coupled::<BogackiShampineTableau>(
                    AppliedLoad::UniaxialStress(load, &times),
                    NewtonRaphson::default(),
                )
                .unwrap();
            let (_, split, _) = model()
                .root_rkmk::<BogackiShampineTableau>(
                    AppliedLoad::UniaxialStress(load, &times),
                    NewtonRaphson::default(),
                )
                .unwrap();
            let coupled = coupled.iter().last().unwrap().clone();
            let signed = |deformation_gradient: &DeformationGradient| {
                deformation_gradient[1][1].value() - reference[1][1].value()
            };
            let (lead, lag) = (signed(&coupled), signed(split.iter().last().unwrap()));
            println!("{steps}: coupled {lead:e}, split {lag:e}");
            assert!(
                lead < 0.0 && lag > 0.0,
                "the two maps do not straddle the reference at {steps} steps: {lead:e}, {lag:e}"
            );
            errors.push((&coupled - &reference).norm().value());
            let deformation_gradient_p = &state_variables.iter().last().unwrap().0;
            assert!((deformation_gradient_p.determinant() - 1.0).abs() < 1e-10);
            assert!(
                (deformation_gradient_p - &DeformationGradientPlastic::identity())
                    .norm()
                    .value()
                    > 1e-3
            );
        }
        assert!(*errors.last().unwrap() < 1e-3, "not accurate: {errors:?}");
        errors.windows(2).for_each(|pair| {
            let ratio = pair[0] / pair[1];
            assert!(
                (1.7..2.4).contains(&ratio),
                "not first order: {ratio}, {errors:?}"
            )
        });
    }

    // Both frozen-drive maps are first order; resolving F at every stage
    // abscissa lifts the return map to the tableau's own order.
    #[test]
    fn rkmk_dae_is_third_order_where_the_frozen_drive_maps_are_first() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, FirstOrderRoot, RkmkRoot},
            math::{integrate::BogackiShampine, optimize::NewtonRaphson},
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
        let (_, reference, _) = model()
            .root(
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
        for steps in [5, 10, 20, 40] {
            let times = time(steps);
            let (_, dae, state_variables) = model()
                .root_rkmk_dae::<BogackiShampineTableau, Quantity>(
                    AppliedLoad::UniaxialStress(load, &times),
                    NewtonRaphson::default(),
                )
                .unwrap();
            let (_, split, _) = model()
                .root_rkmk::<BogackiShampineTableau>(
                    AppliedLoad::UniaxialStress(load, &times),
                    NewtonRaphson::default(),
                )
                .unwrap();
            let error = (dae.iter().last().unwrap() - &reference).norm().value();
            let error_split = (split.iter().last().unwrap() - &reference).norm().value();
            println!("{steps}: dae {error:e}, split {error_split:e}");
            assert!(
                error < error_split / 100.0,
                "stage-resolved not far better at {steps}: {error:e} vs {error_split:e}"
            );
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

    // The additive DAE root only gets det F_p = 1 by integrating accurately
    // enough -- its drift tracks the tolerance. Reconstructing through `expm`
    // makes it structural instead, at any step size.
    #[test]
    fn rkmk_dae_keeps_the_group_structurally_where_the_additive_root_earns_it() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, FirstOrderRoot},
            math::{Scalar, integrate::BogackiShampine, optimize::NewtonRaphson},
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
        let drift = |tol: Scalar| {
            let (_, _, state_variables) = model()
                .root(
                    AppliedLoad::UniaxialStress(load, &span),
                    BogackiShampine {
                        abs_tol: tol,
                        rel_tol: tol,
                        ..Default::default()
                    },
                    NewtonRaphson::default(),
                )
                .unwrap();
            (state_variables.iter().last().unwrap().0.determinant() - 1.0).abs()
        };
        let (loose, tight) = (drift(1e-4), drift(1e-8));
        println!("additive drift: {loose:e} at 1e-4, {tight:e} at 1e-8");
        assert!(loose > 1e-7, "additive root did not drift: {loose:e}");
        assert!(tight < loose / 100.0, "drift did not track the tolerance");
        let (_, _, state_variables) = model()
            .root_rkmk_dae::<BogackiShampineTableau, Quantity>(
                AppliedLoad::UniaxialStress(load, &time(5)),
                NewtonRaphson::default(),
            )
            .unwrap();
        state_variables
            .iter()
            .for_each(|state| assert!((state.0.determinant() - 1.0).abs() < 1e-13));
    }

    #[test]
    fn rkmk_dae_adaptive_subdivides_and_meets_its_tolerance() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, FirstOrderRoot},
            math::{Scalar, integrate::BogackiShampine, optimize::NewtonRaphson},
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let span = [Quantity::<Time>::new(0.0), Quantity::<Time>::new(1.0)];
        let (_, reference, _) = model()
            .root(
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
        let run = |tol: Scalar| {
            model()
                .root_rkmk_dae_adaptive::<BogackiShampineTableau, Quantity>(
                    AppliedLoad::UniaxialStress(load, &span),
                    NewtonRaphson::default(),
                    tol,
                    tol,
                )
                .unwrap()
        };
        let (times, deformation_gradients, state_variables) = run(1e-9);
        // the controller subdivided the single [0, 1] span
        assert!(times.len() > 2);
        let error = (deformation_gradients.iter().last().unwrap() - &reference)
            .norm()
            .value();
        println!("adaptive: {} steps, error {error:e}", times.len() - 1);
        assert!(error < 1e-6, "tolerance not met: {error:e}");
        state_variables
            .iter()
            .for_each(|state| assert!((state.0.determinant() - 1.0).abs() < 1e-13));
        assert!(
            (&state_variables.iter().last().unwrap().0 - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
        // a much looser tolerance takes fewer steps
        let (loose_times, _, _) = run(1e-4);
        assert!(loose_times.len() < times.len());
    }

    // More than two times request report times rather than a span: the state is
    // interpolated off the accepted steps in the algebra, so it stays on the
    // group there too.
    #[test]
    fn rkmk_dae_adaptive_reports_on_the_group_at_requested_times() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::AppliedLoad, math::optimize::NewtonRaphson,
        };
        let load = |t: Quantity<Time>| 1.0 + t.value();
        let requested = time(13);
        let span = [requested[0], *requested.last().unwrap()];
        // reference: the fixed-step stage-resolved map on a grid 40x finer, whose
        // every 40th sample is a requested time
        let (_, reference, reference_state) = model()
            .root_rkmk_dae::<BogackiShampineTableau, Quantity>(
                AppliedLoad::UniaxialStress(load, &time(13 * 40)),
                NewtonRaphson::default(),
            )
            .unwrap();
        let (times, deformation_gradients, state_variables) = model()
            .root_rkmk_dae_adaptive::<BogackiShampineTableau, Quantity>(
                AppliedLoad::UniaxialStress(load, &requested),
                NewtonRaphson::default(),
                1e-9,
                1e-9,
            )
            .unwrap();
        assert_eq!(times.len(), requested.len());
        times
            .iter()
            .zip(requested.iter())
            .for_each(|(reported, request)| assert_eq!(reported.value(), request.value()));
        // the accepted steps the controller actually took are not the requested ones
        let (accepted, _, _) = model()
            .root_rkmk_dae_adaptive::<BogackiShampineTableau, Quantity>(
                AppliedLoad::UniaxialStress(load, &span),
                NewtonRaphson::default(),
                1e-9,
                1e-9,
            )
            .unwrap();
        assert!(accepted.len() > 4 * requested.len());
        let mut worst = 0.0_f64;
        for (k, (state, deformation_gradient)) in state_variables
            .iter()
            .zip(deformation_gradients.iter())
            .enumerate()
        {
            assert!(
                (state.0.determinant() - 1.0).abs() < 1e-10,
                "off the group at requested time {k}"
            );
            worst = worst
                .max((deformation_gradient - &reference[40 * k]).norm().value())
                .max((&state.0 - &reference_state[40 * k].0).norm().value());
        }
        println!("dense output vs refined reference: {worst:e}");
        assert!(worst < 1e-6, "dense output disagrees: {worst:e}");
        // and the plastic state actually flowed, so none of this is vacuous
        assert!(
            (&state_variables.iter().last().unwrap().0 - &DeformationGradientPlastic::identity())
                .norm()
                .value()
                > 1e-3
        );
    }

    #[test]
    fn rkmk_dae_keeps_the_internal_dissipation_non_negative() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, ElasticViscoplastic},
            math::optimize::NewtonRaphson,
        };
        let model = model();
        let (_, deformation_gradients, state_variables) = model
            .root_rkmk_dae::<BogackiShampineTableau, Quantity>(
                AppliedLoad::UniaxialStress(|t: Quantity<Time>| 1.0 + 2.0 * t.value(), &time(24)),
                NewtonRaphson::default(),
            )
            .unwrap();
        deformation_gradients
            .iter()
            .zip(state_variables.iter())
            .for_each(|(deformation_gradient, state)| {
                Assert::non_negative(
                    &model
                        .internal_dissipation(deformation_gradient, state)
                        .unwrap(),
                )
                .unwrap()
            });
    }

    #[test]
    fn coupled_keeps_the_internal_dissipation_non_negative() {
        use crate::{
            constitutive::solid::elastic_viscoplastic::{AppliedLoad, ElasticViscoplastic},
            math::optimize::NewtonRaphson,
        };
        let times = time(24);
        let model = model();
        let (_, deformation_gradients, state_variables) = model
            .root_rkmk_coupled::<BogackiShampineTableau>(
                AppliedLoad::UniaxialStress(|t: Quantity<Time>| 1.0 + 2.0 * t.value(), &times),
                NewtonRaphson::default(),
            )
            .unwrap();
        deformation_gradients
            .iter()
            .zip(state_variables.iter())
            .for_each(|(deformation_gradient, state)| {
                assert!((state.0.determinant() - 1.0).abs() < 1e-10);
                Assert::non_negative(
                    &model
                        .internal_dissipation(deformation_gradient, state)
                        .unwrap(),
                )
                .unwrap()
            });
    }
}
