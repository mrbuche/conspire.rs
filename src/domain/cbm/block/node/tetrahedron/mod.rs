use crate::{
    math::{Quantity, Reference, TensorRank1List},
    mechanics::ReferenceCoordinates,
    units::{ReciprocalLength, Volume},
};

pub(crate) type ElementNodalReferenceCoordinates = ReferenceCoordinates<4>;
pub(crate) type GradientVectors = TensorRank1List<3, Reference, 4, ReciprocalLength>;

#[derive(Clone, Debug)]
pub(crate) struct Tetrahedron {
    gradient_vectors: GradientVectors,
    volume: Quantity<Volume>,
}

impl Tetrahedron {
    pub(crate) fn gradient_vectors(&self) -> &GradientVectors {
        &self.gradient_vectors
    }
    pub(crate) fn volume(&self) -> Quantity<Volume> {
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
