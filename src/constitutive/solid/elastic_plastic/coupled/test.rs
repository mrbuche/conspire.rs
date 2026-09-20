use super::{Iterate, SIZE, Sensitivities, Unknowns};
use crate::{
    constitutive::{
        ConstitutiveError,
        canonical::Canonical,
        fluid::plastic::{Plastic, PlasticFlow, RateIndependentPlastic},
        solid::{
            elastic_plastic::{ElasticPlastic, ElasticPlasticOrViscoplastic},
            hyperelastic::{Hencky, NeoHookean, SaintVenantKirchhoff},
        },
    },
    math::Quantity,
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::Stress,
};

fn stretch_shear(s: f64) -> DeformationGradient {
    DeformationGradient::from([
        [1.0 + s, 0.7 * s, 0.2 * s],
        [0.0, 1.0 - 0.2 * s, 0.4 * s],
        [0.0, 0.0, 1.0 + 0.3 * s],
    ])
}

macro_rules! test_models {
    ($elastic:ident) => {
        use super::*;

        fn model() -> Canonical<$elastic, PlasticFlow> {
            Canonical::from((
                $elastic {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                PlasticFlow {
                    yield_stress: Stress::pascals(2.0),
                    hardening_slope: Stress::pascals(1.0),
                },
            ))
        }

        /// The analytic Jacobian of the coupled residual against central differences at
        /// an iterate whose plastic increment is not symmetric, so the nine-component
        /// parametrization is exercised beyond the symmetric trace-free subspace.
        #[test]
        fn jacobian_matches_finite_difference() -> Result<(), ConstitutiveError> {
            let model = model();
            let f = stretch_shear(0.4);
            let state = model.initial_state();
            let (f_p_n, &strain_n): (&DeformationGradientPlastic, &Quantity) = (&state).into();
            let strain_n = strain_n.value();
            let x: Unknowns = [
                0.030, 0.010, 0.004, 0.012, -0.020, 0.005, 0.003, -0.002, -0.010, 0.030,
            ];
            let iterate = Iterate::new(&model, &f, f_p_n, strain_n, &x)?;
            let analytic = Sensitivities::new(&model, &f, f_p_n, &x, &iterate)?
                .jacobian(x[9], model.hardening_slope().value());
            let h = 1e-7;
            for column in 0..SIZE {
                let (mut plus, mut minus) = (x, x);
                plus[column] += h;
                minus[column] -= h;
                let r_plus = Iterate::new(&model, &f, f_p_n, strain_n, &plus)?.residual;
                let r_minus = Iterate::new(&model, &f, f_p_n, strain_n, &minus)?.residual;
                for row in 0..SIZE {
                    let finite_difference = (r_plus[row] - r_minus[row]) / (2.0 * h);
                    assert!(
                        (analytic[row][column] - finite_difference).abs()
                            <= 1e-6 * (1.0 + analytic[row][column].abs()),
                        "J[{row}][{column}]: analytic {} vs finite difference {finite_difference}",
                        analytic[row][column],
                    );
                }
            }
            Ok(())
        }

        /// The consistent tangent against central differences of the stress through the
        /// whole return map, from a virgin state, on a large step, and from a
        /// pre-loaded state under a rotated load.
        #[test]
        fn consistent_tangent_matches_finite_difference() -> Result<(), ConstitutiveError> {
            let model = model();
            let initial = model.initial_state();
            let pre_loaded = model.return_map(&stretch_shear(0.5), &initial)?;
            let rotated =
                DeformationGradient::from([[1.55, 0.5, 0.1], [0.2, 0.95, 0.3], [-0.1, 0.05, 1.1]]);
            let cases = [
                (stretch_shear(0.3), initial.clone()),
                (stretch_shear(0.6), initial),
                (rotated, pre_loaded),
            ];
            let h = 1e-4;
            for (case, (f, state)) in cases.iter().enumerate() {
                let (tangent, _) = model.consistent_tangent_stiffness(f, state)?;
                for k in 0..3 {
                    for l in 0..3 {
                        let stress_at = |sign: f64| -> Result<_, ConstitutiveError> {
                            let mut perturbed = f.clone();
                            perturbed[k][l] += Quantity::new(sign * h);
                            let updated = model.return_map(&perturbed, state)?;
                            model.first_piola_kirchhoff_stress(&perturbed, &updated.0)
                        };
                        let finite_difference = (stress_at(1.0)? - stress_at(-1.0)?) / (2.0 * h);
                        for i in 0..3 {
                            for j in 0..3 {
                                let analytic = tangent[i][j][k][l].value();
                                assert!(
                                    (analytic - finite_difference[i][j].value()).abs()
                                        <= 1e-5 * (1.0 + analytic.abs()),
                                    "case {case}, tangent[{i}][{j}][{k}][{l}]: analytic \
                                     {analytic} vs finite difference {}",
                                    finite_difference[i][j].value(),
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    };
}

mod hencky {
    test_models!(Hencky);
}

mod neo_hookean {
    test_models!(NeoHookean);
}

mod saint_venant_kirchhoff {
    test_models!(SaintVenantKirchhoff);
}
