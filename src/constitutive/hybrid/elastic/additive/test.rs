use crate::constitutive::hybrid::{
    elastic::test::test_hybrid_elastic_constitutive_models, Additive,
};

test_hybrid_elastic_constitutive_models!(Additive);
