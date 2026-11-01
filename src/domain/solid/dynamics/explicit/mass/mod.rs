//! The masses of a run of explicit dynamics.

use crate::domain::{
    Model,
    solid::mass::{
        ConsistentMassElements, LumpedMassElements, MassMatrix, NodalLumpedMasses, NodalMasses,
    },
};

/// Which masses a run of explicit dynamics uses, taken from the blocks of the model.
///
/// The model makes the masses from the same densities that the estimate of the stability limit
/// uses, so the two cannot disagree, and none are passed in.
pub trait MassChoice<M> {
    /// The masses of the choice.
    type Mass: MassMatrix<3>;
    /// The masses of the model.
    fn masses(self, model: &M) -> Self::Mass;
}

/// The lumped masses of the model, which need a division to give the accelerations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lumped;

/// The consistent masses of the model, which need a solve to give the accelerations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Consistent;

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
