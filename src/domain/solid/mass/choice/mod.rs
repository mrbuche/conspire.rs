//! The choice of the masses of a model, lumped or consistent, that a load or a run asks for.

use super::{
    ConsistentMassElements, InertialMass, LumpedMassElements, NodalLumpedMasses, NodalMasses,
};
use crate::domain::Model;

pub use crate::domain::lumping::{Consistent, Lumped};

/// Which masses are used, taken from the blocks of the model.
///
/// The model makes the masses from the same densities that the estimate of the stability limit
/// uses, so the two cannot disagree, and none are passed in.
pub trait MassChoice<M> {
    /// The masses of the choice.
    type Mass: InertialMass<3>;
    /// The masses of the model.
    fn masses(self, model: &M) -> Self::Mass;
}

impl<B> MassChoice<Model<B, 3>> for Lumped
where
    B: LumpedMassElements<3>,
{
    type Mass = NodalLumpedMasses;
    fn masses(self, model: &Model<B, 3>) -> NodalLumpedMasses {
        model.nodal_lumped_masses()
    }
}

impl<B> MassChoice<Model<B, 3>> for Consistent
where
    B: ConsistentMassElements,
{
    type Mass = NodalMasses;
    fn masses(self, model: &Model<B, 3>) -> NodalMasses {
        model.nodal_masses()
    }
}
