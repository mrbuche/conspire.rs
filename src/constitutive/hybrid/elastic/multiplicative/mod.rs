mod elastic;
mod viscoplastic;

use crate::{
    constitutive::{
        fluid::viscoplastic::Viscoplastic, hybrid::Multiplicative, solid::elastic::Elastic,
    },
    math::Tensor,
};
use std::{marker::PhantomData, ops::Deref};

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

/// A hybrid elastic-viscoplastic constitutive model based on the multiplicative decomposition.
#[derive(Clone, Debug)]
pub struct ElasticMultiplicativeViscoplastic<C1, C2, Y2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Tensor,
{
    inner: Multiplicative<C1, C2>,
    dummy: PhantomData<Y2>,
}

impl<C1, C2, Y2> Deref for ElasticMultiplicativeViscoplastic<C1, C2, Y2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Tensor,
{
    type Target = Multiplicative<C1, C2>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C1, C2, Y2> From<(C1, C2)> for ElasticMultiplicativeViscoplastic<C1, C2, Y2>
where
    C1: Elastic,
    C2: Viscoplastic<Y2>,
    Y2: Tensor,
{
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self {
            inner: Multiplicative(constitutive_model_1, constitutive_model_2),
            dummy: PhantomData,
        }
    }
}
