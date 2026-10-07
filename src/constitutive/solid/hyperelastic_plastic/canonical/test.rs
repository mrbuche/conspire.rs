use crate::{
    EPSILON,
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{Linear, PlasticFlow, PlasticWork, VonMises, YieldSurface},
        solid::{
            elastic_plastic::{AppliedLoad, ElasticPlasticOrViscoplastic, FirstOrderRoot},
            hyperelastic::{Hencky, NeoHookean, SaintVenantKirchhoff},
            hyperelastic_plastic::HyperelasticPlastic,
        },
    },
    math::{
        Quantity, Rank2, Tensor,
        assert::{Assert, AssertionError, perturbation},
        optimize::{NewtonRaphson, SolveStrategy},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::{Stress, Time},
};

macro_rules! test_canonical {
    ($elastic:ident) => {
        use super::*;
        fn model() -> Canonical<$elastic, PlasticFlow<VonMises, Linear>> {
            Canonical::from((
                $elastic {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                PlasticFlow {
                    surface: VonMises,
                    hardening: Linear {
                        yield_stress: Stress::pascals(2.0),
                        hardening_slope: Stress::pascals(1.0),
                    },
                },
            ))
        }
        fn deformation_gradient_p() -> DeformationGradientPlastic {
            DeformationGradientPlastic::from([
                [1.2, 0.1, 0.05],
                [0.0, 0.5, 0.2],
                [0.0, 0.0, 1.0 / 0.6],
            ])
        }
        fn deformation_gradient() -> DeformationGradient {
            DeformationGradient::from([
                [1.31924942, 0.36431217, 0.41764434],
                [0.09959341, 1.08409741, 0.48320137],
                [0.21114106, 0.16675104, 1.18146028],
            ])
        }
        #[test]
        fn stress_is_energy_gradient() -> Result<(), AssertionError> {
            let model = model();
            let (f, f_p) = (deformation_gradient(), deformation_gradient_p());
            let stress = model.first_piola_kirchhoff_stress(&f, &f_p)?;
            for i in 0..3 {
                for j in 0..3 {
                    let mut plus = f.clone();
                    plus[i][j] += perturbation(0.5 * EPSILON);
                    let mut minus = f.clone();
                    minus[i][j] -= perturbation(0.5 * EPSILON);
                    let fd = (model.helmholtz_free_energy_density(&plus, &f_p)?
                        - model.helmholtz_free_energy_density(&minus, &f_p)?)
                    .value()
                        / EPSILON;
                    let exact = stress[i][j].value();
                    assert!(
                        (fd - exact).abs() <= 1e-5 * exact.abs().max(1.0),
                        "dPsi/dF[{i}][{j}]: fd {fd} vs P {exact}",
                    );
                }
            }
            Ok(())
        }
        #[test]
        fn plastic_flow_releases_mandel_power() -> Result<(), AssertionError> {
            let model = model();
            let (f, f_p) = (deformation_gradient(), deformation_gradient_p());
            let deviatoric = model.mandel_stress(&f, &f_p)?.deviatoric();
            let direction = {
                let direction = model.flow_direction(&deviatoric)?;
                (&direction + direction.transpose()) * 0.5
            };
            let step = 1e-6;
            let energy = |gamma: f64| -> Result<f64, AssertionError> {
                let f_p_gamma = (&direction * gamma).expm().unwrap() * &f_p;
                Ok(model.helmholtz_free_energy_density(&f, &f_p_gamma)?.value())
            };
            let fd = (energy(step)? - energy(-step)?) / (2.0 * step);
            let exact = -deviatoric.norm().value();
            assert!(
                (fd - exact).abs() <= 1e-5 * exact.abs().max(1.0),
                "da/dgamma: fd {fd} vs -|M'| {exact}",
            );
            Ok(())
        }
        #[test]
        fn solved_load_path_balances_energy() -> Result<(), AssertionError> {
            let model = model();
            let steps: Vec<Quantity<Time>> = (0..=400)
                .map(|step| Quantity::new(0.5 * step as f64 / 400.0))
                .collect();
            let (_, fs, states) = FirstOrderRoot::root(
                &model,
                AppliedLoad::UniaxialStress(|t: Quantity<Time>| 1.0 + t.value(), &steps),
                NewtonRaphson::default(),
                SolveStrategy::Condensed(NewtonRaphson::default()),
            )?;
            let (fs, states) = (fs.as_slice(), states.as_slice());
            let stresses = fs
                .iter()
                .zip(states)
                .map(|(f, state)| model.first_piola_kirchhoff_stress(f, &state.0))
                .collect::<Result<Vec<_>, _>>()?;
            let work: Quantity<Stress> = stresses
                .windows(2)
                .zip(fs.windows(2))
                .map(|(p, f)| {
                    ((p[0].clone() + &p[1]) * 0.5 * (f[1].clone() - &f[0]).transpose()).trace()
                })
                .sum();
            let dissipation = model.plastic_work_density(states[states.len() - 1].1)?
                - model.plastic_work_density(states[0].1)?;
            let energy = model
                .helmholtz_free_energy_density(&fs[fs.len() - 1], &states[states.len() - 1].0)?
                - model.helmholtz_free_energy_density(&fs[0], &states[0].0)?;
            assert!(
                dissipation.value() > 1e-2,
                "the load path must yield for this test to mean anything"
            );
            Assert {
                rel_tol: 2e-3,
                ..Default::default()
            }
            .eq_within_tols(work, &(energy + dissipation))
        }
    };
}

mod neo_hookean {
    test_canonical!(NeoHookean);
}

mod hencky {
    test_canonical!(Hencky);
}

mod saint_venant_kirchhoff {
    test_canonical!(SaintVenantKirchhoff);
}
