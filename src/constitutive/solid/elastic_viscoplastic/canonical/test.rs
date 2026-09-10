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
        math::{
            Quantity, Tensor, TensorArray, TensorTuple, TensorVector,
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
}
