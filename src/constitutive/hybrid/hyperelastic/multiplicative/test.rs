use crate::constitutive::hybrid::{
    hyperelastic::test::test_hybrid_hyperelastic_constitutive_models_no_tangents, Multiplicative,
};

test_hybrid_hyperelastic_constitutive_models_no_tangents!(Multiplicative);
