use crate::constitutive::hybrid::{
    Additive, elastic::test::test_hybrid_elastic_constitutive_models,
};

test_hybrid_elastic_constitutive_models!(Additive);
