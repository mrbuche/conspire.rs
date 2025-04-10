use crate::constitutive::hybrid::{
    Multiplicative, hyperelastic::test::test_hybrid_hyperelastic_constitutive_models_no_tangents,
};

test_hybrid_hyperelastic_constitutive_models_no_tangents!(Multiplicative);
