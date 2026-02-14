macro_rules! test_hybrid_hyperelastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::constitutive::solid::hyperelastic::{ArrudaBoyce, Fung, test::*};
        test_solid_hyperelastic_constitutive_model!($hybrid_type::from((
            ArrudaBoyce {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
                number_of_links: NUMBER_OF_LINKS,
            },
            Fung {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
                exponent: EXPONENT,
                extra_modulus: EXTRA_MODULUS,
            }
        )));
    };
}
pub(crate) use test_hybrid_hyperelastic_constitutive_models;
