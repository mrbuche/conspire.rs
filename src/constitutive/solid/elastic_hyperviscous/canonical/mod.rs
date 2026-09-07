#[cfg(test)]
mod test;

use crate::constitutive::{
    canonical::Canonical,
    fluid::hyperviscous::Hyperviscous,
    solid::{elastic::Elastic, elastic_hyperviscous::ElasticHyperviscous},
};

impl<C1, C2> ElasticHyperviscous for Canonical<C1, C2>
where
    C1: Elastic,
    C2: Hyperviscous,
{
}
