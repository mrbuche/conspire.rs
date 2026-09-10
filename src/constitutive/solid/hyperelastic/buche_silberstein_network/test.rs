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

mod radial_kernel_fit {
    use super::*;
    use crate::math::special::extensible_langevin;

    /// The Chebyshev fit of `G_a` must match the live Gauss-Laguerre quadrature
    /// over the link stiffnesses and Gaussian widths the model reaches
    /// (`w = k1 N_b s / 2`, `ln w` roughly in `[-4, 12]` for any realistic
    /// deformation). Consistency of the derived stress kernel `G` with this fit
    /// is covered by the free-energy finite-difference test.
    #[test]
    fn matches_live_quadrature() {
        for kappa in [1.0, 3.0, 10.0, 50.0] {
            let kernels = RadialKernels::get(kappa);
            for i in 0..40 {
                let w = (-4.0 + i as f64 * 0.4_f64).exp();
                let energy = radial_moment(w, 2.0, |lambda| {
                    extensible_langevin::helmholtz_free_energy(lambda, kappa)
                });
                let relative = (kernels.radial_energy(w) - energy).abs() / energy.abs();
                assert!(
                    relative < 5e-8,
                    "kappa={kappa} w={w:e}: G_a rel {relative:e}"
                );
            }
        }
    }
}

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
            number_of_links: 2000.0,
            link_stiffness: LINK_STIFFNESS,
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
