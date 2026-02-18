use crate::constitutive::hybrid::{
    ElasticAdditive, hyperelastic::test::test_hybrid_hyperelastic_constitutive_models,
};

test_hybrid_hyperelastic_constitutive_models!(ElasticAdditive);
