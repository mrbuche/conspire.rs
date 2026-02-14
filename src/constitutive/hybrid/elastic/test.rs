macro_rules! test_hybrid_elastic_constitutive_models {
    ($hybrid_type: ident) => {
        use crate::constitutive::solid::{
            elastic::{AlmansiHamel, test::*},
            hyperelastic::NeoHookean,
        };
        test_solid_elastic_constitutive_model!($hybrid_type::from((
            AlmansiHamel {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
            NeoHookean {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
        )));
    };
}
pub(crate) use test_hybrid_elastic_constitutive_models;
