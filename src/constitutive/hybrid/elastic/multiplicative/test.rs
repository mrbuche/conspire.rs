use crate::constitutive::hybrid::{
    Multiplicative, elastic::test::test_hybrid_elastic_constitutive_models_no_tangents,
};

test_hybrid_elastic_constitutive_models_no_tangents!(Multiplicative);
