use crate::constitutive::hybrid::{
    hyperelastic::test::test_hybrid_hyperelastic_constitutive_models, Additive,
};

test_hybrid_hyperelastic_constitutive_models!(Additive);
