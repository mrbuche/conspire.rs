#[cfg(test)]
mod test;

use crate::constitutive::{
    canonical::Canonical,
    fluid::plastic::RateIndependentPlastic,
    solid::{elastic::Elastic, elastic_plastic::ElasticPlastic},
};

impl<C1, C2> RateIndependentPlastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: RateIndependentPlastic,
{
}

impl<C1, C2> ElasticPlastic for Canonical<C1, C2>
where
    C1: Elastic,
    C2: RateIndependentPlastic,
{
}
