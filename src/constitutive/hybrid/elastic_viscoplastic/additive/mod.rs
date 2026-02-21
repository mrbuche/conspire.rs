mod elastic;
mod viscoplastic;

use crate::{
    constitutive::{
        fluid::viscoplastic::Viscoplastic,
        hybrid::Additive,
        solid::{elastic::Elastic, elastic_viscoplastic::ElasticViscoplastic},
    },
    math::Tensor,
};
use std::{marker::PhantomData, ops::Deref};

/// A hybrid elastic-viscoplastic constitutive model based on the additive decomposition.
#[derive(Clone, Debug)]
pub struct ElasticViscoplasticAdditiveElastic<C1, C2, Y1>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Elastic,
    Y1: Tensor,
{
    inner: Additive<C1, C2>,
    dummy: PhantomData<Y1>,
}

impl<C1, C2, Y1> Deref for ElasticViscoplasticAdditiveElastic<C1, C2, Y1>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Elastic,
    Y1: Tensor,
{
    type Target = Additive<C1, C2>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C1, C2, Y1> From<(C1, C2)> for ElasticViscoplasticAdditiveElastic<C1, C2, Y1>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Elastic,
    Y1: Tensor,
{
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self {
            inner: Additive(constitutive_model_1, constitutive_model_2),
            dummy: PhantomData,
        }
    }
}

/// A hybrid viscoplastic constitutive model based on the additive decomposition.
#[derive(Clone, Debug)]
pub struct ElasticViscoplasticAdditiveViscoplastic<C1, C2, Y1, Y2>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Viscoplastic<Y2>,
    Y1: Tensor,
    Y2: Tensor,
{
    inner: Additive<C1, C2>,
    dum_1: PhantomData<Y1>,
    dum_2: PhantomData<Y2>,
}

impl<C1, C2, Y1, Y2> Deref for ElasticViscoplasticAdditiveViscoplastic<C1, C2, Y1, Y2>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Viscoplastic<Y2>,
    Y1: Tensor,
    Y2: Tensor,
{
    type Target = Additive<C1, C2>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<C1, C2, Y1, Y2> From<(C1, C2)> for ElasticViscoplasticAdditiveViscoplastic<C1, C2, Y1, Y2>
where
    C1: ElasticViscoplastic<Y1>,
    C2: Viscoplastic<Y2>,
    Y1: Tensor,
    Y2: Tensor,
{
    fn from((constitutive_model_1, constitutive_model_2): (C1, C2)) -> Self {
        Self {
            inner: Additive(constitutive_model_1, constitutive_model_2),
            dum_1: PhantomData,
            dum_2: PhantomData,
        }
    }
}
