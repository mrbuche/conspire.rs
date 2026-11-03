//! The choice of the heat capacities of a model, lumped or consistent, that a run asks for.

use super::{
    ConsistentHeatCapacityElements, InertialHeatCapacity, LumpedHeatCapacityElements,
    NodalHeatCapacities, NodalLumpedHeatCapacities,
};
use crate::domain::{
    Model,
    lumping::{Consistent, Lumped},
};

/// Which heat capacities a run uses, taken from the blocks of the model.
///
/// The model makes the heat capacities from the same densities that the estimate of the
/// stability limit uses, so the two cannot disagree, and none are passed in.
pub trait CapacityChoice<M> {
    /// The heat capacities of the choice.
    type Capacity: InertialHeatCapacity;
    /// The heat capacities of the model.
    fn capacities(self, model: &M) -> Self::Capacity;
}

impl<B, const D: usize> CapacityChoice<Model<B, D>> for Lumped
where
    B: LumpedHeatCapacityElements<D>,
{
    type Capacity = NodalLumpedHeatCapacities;
    fn capacities(self, model: &Model<B, D>) -> NodalLumpedHeatCapacities {
        model.nodal_lumped_heat_capacities()
    }
}

impl<B, const D: usize> CapacityChoice<Model<B, D>> for Consistent
where
    B: ConsistentHeatCapacityElements,
{
    type Capacity = NodalHeatCapacities;
    fn capacities(self, model: &Model<B, D>) -> NodalHeatCapacities {
        model.nodal_heat_capacities()
    }
}
