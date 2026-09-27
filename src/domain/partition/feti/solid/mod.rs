use super::block::element::{DecomposableElements, ElementSystems};
use crate::domain::{ElementModelError, Model, NodalCoordinates, ProvidesTangent};

impl<B> ProvidesTangent<NodalCoordinates<3>, ElementSystems> for Model<B, 3>
where
    B: DecomposableElements,
{
    fn provide_tangent(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        self.element_systems(nodal_coordinates)
    }
}

#[cfg(feature = "fem")]
use super::Feti;
#[cfg(feature = "fem")]
use crate::{
    domain::{SolverFor, solid::NodalForcesSolid},
    math::{Quantity, optimize::NewtonRaphson},
    units::Energy,
};

#[cfg(feature = "fem")]
impl<B> SolverFor<Model<B, 3>, Quantity<Energy>, NodalForcesSolid<3>> for NewtonRaphson<Feti>
where
    B: DecomposableElements,
{
    type Tangent = ElementSystems;
    const SPARSE: bool = false;
}
