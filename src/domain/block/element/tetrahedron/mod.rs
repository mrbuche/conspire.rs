use crate::{
    math::{Quantity, Reference, TensorRank1List},
    mechanics::ReferenceCoordinates,
    units::{ReciprocalLength, Volume},
};

pub type ElementNodalReferenceCoordinates = ReferenceCoordinates<4>;
pub type GradientVectors = TensorRank1List<3, Reference, 4, ReciprocalLength>;

/// A linear tetrahedron's reference-configuration gradient vectors and volume
/// — the one piece of fem's element machinery vem/cbm actually need, kept
/// here so they don't have to pull in fem's generic `FiniteElement<G,M,N,P,W>`
/// machinery just for this.
#[derive(Clone, Debug)]
pub struct Tetrahedron {
    gradient_vectors: GradientVectors,
    volume: Quantity<Volume>,
}

impl Tetrahedron {
    pub fn gradient_vectors(&self) -> &GradientVectors {
        &self.gradient_vectors
    }
    pub fn volume(&self) -> Quantity<Volume> {
        self.volume
    }
}

impl From<ElementNodalReferenceCoordinates> for Tetrahedron {
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates) -> Self {
        let standard_gradient_operator: TensorRank1List<3, Reference, 4> = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into();
        let jacobian = &reference_nodal_coordinates * &standard_gradient_operator;
        let volume = Quantity::new(jacobian.determinant() / 6.0);
        let gradient_vectors = jacobian.inverse_transpose() * &standard_gradient_operator;
        Self {
            gradient_vectors,
            volume,
        }
    }
}
