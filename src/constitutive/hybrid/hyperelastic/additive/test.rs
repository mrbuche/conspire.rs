use crate::constitutive::hybrid::{
    Additive, hyperelastic::test::test_hybrid_hyperelastic_constitutive_models,
};

test_hybrid_hyperelastic_constitutive_models!(Additive);
