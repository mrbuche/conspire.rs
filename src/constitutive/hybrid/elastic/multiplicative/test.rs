use crate::constitutive::hybrid::{
    elastic::test::test_hybrid_elastic_constitutive_models_no_tangents, Multiplicative,
};

test_hybrid_elastic_constitutive_models_no_tangents!(Multiplicative);
