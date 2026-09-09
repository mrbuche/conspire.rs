// `use_elastic_macros!` also pulls in the tangent-stress helper macros, which
// this model does not exercise while its tangent stiffness is `todo!`.
#![allow(unused_imports)]

use super::super::test::*;
use super::*;

const LINK_STIFFNESS: Scalar = 50.0;

use_elastic_macros!();

test_solid_hyperelastic_constitutive_model_no_tangents!(BucheSilberstein {
    bulk_modulus: BULK_MODULUS,
    shear_modulus: SHEAR_MODULUS,
    number_of_links: NUMBER_OF_LINKS,
    link_stiffness: LINK_STIFFNESS,
});

mod reduced_efjc_force_inverse {
    use super::*;
    use crate::math::special::langevin;

    #[test]
    fn inverts_the_reduced_relation() {
        let model = BucheSilberstein {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
            number_of_links: NUMBER_OF_LINKS,
            link_stiffness: LINK_STIFFNESS,
        };
        for &gamma in &[1e-3, 0.1, 0.35, 0.7, 0.999, 1.5, 4.0] {
            let eta = model.nondimensional_force(gamma);
            let residual = langevin(eta) + eta / LINK_STIFFNESS - gamma;
            assert!(
                residual.abs() < 1e-9,
                "gamma = {gamma}: residual = {residual:e}"
            );
        }
    }
}
