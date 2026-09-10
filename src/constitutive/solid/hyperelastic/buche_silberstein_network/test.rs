#![allow(unused_imports)]

use super::super::test::*;
use super::*;

const LINK_STIFFNESS: Scalar = 50.0;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model_no_tangents!(BucheSilbersteinNetwork {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    number_of_links: NUMBER_OF_LINKS,
    link_stiffness: LINK_STIFFNESS,
});

mod neo_hookean_limit {
    use super::*;
    use crate::constitutive::solid::hyperelastic::NeoHookean;

    #[test]
    fn many_links_approach_neo_hookean_uniaxial() {
        let deformation_gradient = DeformationGradient::from([
            [1.4, 0.0, 0.0],
            [0.0, 1.0 / 1.4_f64.sqrt(), 0.0],
            [0.0, 0.0, 1.0 / 1.4_f64.sqrt()],
        ]);
        let network = BucheSilbersteinNetwork {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            number_of_links: 800.0,
            link_stiffness: 1e4,
        };
        let neo_hookean = NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        };
        let got = network.cauchy_stress(&deformation_gradient).unwrap();
        let reference = neo_hookean.cauchy_stress(&deformation_gradient).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let (a, b) = (got[i][j].value(), reference[i][j].value());
                assert!(
                    (a - b).abs() < 5e-3 * (1.0 + b.abs()),
                    "[{i}][{j}]: {a} vs {b}"
                );
            }
        }
    }
}
