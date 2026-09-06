use crate::{
    constitutive::solid::{
        elastic::Elastic,
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    mechanics::test::get_deformation_gradient,
    units::Stress,
};

const BULK_MODULUS: f64 = 1.3;
const SHEAR_MODULUS: f64 = 0.7;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

macro_rules! model_tests {
    ($module: ident, $model: ident) => {
        mod $module {
            use super::*;
            use crate::constitutive::autodiff::$module::{
                first_piola_kirchhoff_stress, first_piola_kirchhoff_tangent_stiffness,
            };
            fn model() -> $model {
                $model {
                    bulk_modulus: Stress::pascals(BULK_MODULUS),
                    shear_modulus: Stress::pascals(SHEAR_MODULUS),
                }
            }
            #[test]
            fn stress_matches_hand_written() {
                let (model, f) = (model(), get_deformation_gradient());
                let ad = first_piola_kirchhoff_stress(&model, &f);
                let hand = match model.first_piola_kirchhoff_stress(&f) {
                    Ok(stress) => stress,
                    Err(_) => panic!("hand-written stress failed"),
                };
                for i in 0..3 {
                    for j in 0..3 {
                        assert!(close(ad[i][j].value(), hand[i][j].value(), 1e-8));
                    }
                }
            }
            #[test]
            fn tangent_matches_hand_written() {
                let (model, f) = (model(), get_deformation_gradient());
                let ad = first_piola_kirchhoff_tangent_stiffness(&model, &f);
                let hand = match model.first_piola_kirchhoff_tangent_stiffness(&f) {
                    Ok(tangent) => tangent,
                    Err(_) => panic!("hand-written tangent failed"),
                };
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            for l in 0..3 {
                                assert!(close(
                                    ad[i][j][k][l].value(),
                                    hand[i][j][k][l].value(),
                                    1e-6
                                ));
                            }
                        }
                    }
                }
            }
        }
    };
}

model_tests!(neo_hookean, NeoHookean);
model_tests!(saint_venant_kirchhoff, SaintVenantKirchhoff);
