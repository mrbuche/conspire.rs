mod elastic;

use crate::constitutive::{hybrid::Additive, solid::elastic::Elastic};
use std::ops::Deref;

/// A hybrid elastic constitutive model based on the additive decomposition.
#[derive(Clone, Debug)]
pub struct ElasticAdditive<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    inner: Additive<C1, C2>,
}

impl<C1, C2> Deref for ElasticAdditive<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    type Target = Additive<C1, C2>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C1, C2> From<(C1, C2)> for ElasticAdditive<C1, C2>
where
    C1: Elastic,
    C2: Elastic,
{
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self {
            inner: Additive(constitutive_model_1, constitutive_model_2),
        }
    }
}
