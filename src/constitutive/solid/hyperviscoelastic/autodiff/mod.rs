use crate::constitutive::{
    canonical::Canonical,
    fluid::hyperviscous::autodiff::AutodiffHyperviscous,
    solid::{
        elastic::autodiff::AutodiffElastic,
        hyperelastic::autodiff::{Autodiff, AutodiffHyperelastic},
        hyperviscoelastic::Hyperviscoelastic,
    },
};
use std::fmt::Debug;

pub trait AutodiffHyperviscoelastic
where
    Self: Hyperviscoelastic,
{
    type Elastic: AutodiffHyperelastic;
    type Viscous: AutodiffHyperviscous;
    fn elastic_parameters(&self) -> [f64; 2];
    fn viscous_parameters(&self) -> [f64; 2];
}

impl<E, V> AutodiffHyperviscoelastic for Canonical<Autodiff<E>, Autodiff<V>>
where
    E: AutodiffHyperelastic + Clone + Debug,
    V: AutodiffHyperviscous + Clone + Debug,
    Self: Hyperviscoelastic,
{
    type Elastic = E;
    type Viscous = V;
    fn elastic_parameters(&self) -> [f64; 2] {
        AutodiffElastic::parameters(&self.0.0)
    }
    fn viscous_parameters(&self) -> [f64; 2] {
        self.1.0.parameters()
    }
}
