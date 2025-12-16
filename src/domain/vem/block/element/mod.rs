pub mod solid;

use crate::math::{Scalars, TensorRank1Vec2D};
use std::fmt::{self, Debug, Formatter};

pub type ElementNodalCoordinates = TensorRank1Vec2D<3, 1>;
pub type ElementNodalVelocities = TensorRank1Vec2D<3, 1>;
pub type ElementNodalReferenceCoordinates = TensorRank1Vec2D<3, 0>;

pub struct Element {
    // gradient_vectors: GradientVectors<G, N>,
    integration_weights: Scalars,
}

impl Element {
    // fn gradient_vectors(&self) -> &GradientVectors<G, N> {
    //     &self.gradient_vectors
    // }
    fn integration_weights(&self) -> &Scalars {
        &self.integration_weights
    }
}

impl Debug for Element {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "VirtualElement {{ ... }}",)
    }
}

pub trait VirtualElement
where
    Self: From<ElementNodalReferenceCoordinates>,
{
    // fn initialize(
    //     reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>,
    // ) -> (GradientVectors<G, N>, ScalarList<G>);
}
