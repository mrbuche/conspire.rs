mod elastic;
mod viscoplastic;

use crate::constitutive::{hybrid::Multiplicative, solid::elastic::Elastic};
use std::ops::Deref;

/// A hybrid elastic constitutive model based on the multiplicative decomposition.
#[derive(Clone, Debug)]
pub struct ElasticMultiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    inner: Multiplicative<C1, C2>,
}

impl<C1, C2> Deref for ElasticMultiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    type Target = Multiplicative<C1, C2>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C1, C2> From<(C1, C2)> for ElasticMultiplicative<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self {
            inner: Multiplicative(constitutive_model_1, constitutive_model_2),
        }
    }
}
