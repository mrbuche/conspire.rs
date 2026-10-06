use crate::{
    domain::{Blocks, ElementModelError, Model, NodalCoordinates, block::element::Elements},
    math::Quantity,
    units::Time,
};

/// Elements that report how fast their motion can be.
pub trait TimeScaleElements<const D: usize>
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given coordinates, the
    /// reciprocal of the highest angular frequency, which bounds a stable explicit time step.
    fn fastest_time_scale(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> TimeScaleElements<D> for Model<B, D>
where
    B: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks.fastest_time_scale(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> TimeScaleElements<D> for Blocks<B1, B2>
where
    B1: TimeScaleElements<D>,
    B2: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_time_scale(nodal_coordinates)?
            .min(self.1.fastest_time_scale(nodal_coordinates)?))
    }
}
