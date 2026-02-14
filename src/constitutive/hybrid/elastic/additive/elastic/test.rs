use crate::constitutive::hybrid::{
    ElasticAdditive, elastic::test::test_hybrid_elastic_constitutive_models,
};

test_hybrid_elastic_constitutive_models!(ElasticAdditive);
