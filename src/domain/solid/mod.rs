pub(crate) mod elastic;
pub(crate) mod elastic_hyperviscous;
pub(crate) mod hyperelastic;
pub(crate) mod hyperviscoelastic;
pub(crate) mod viscoelastic;

use crate::{
    domain::{NodalCoordinates, NodalVelocities},
    math::{Current, TensorRank1Vec, TensorRank2SparseVec2D, TensorRank2SparseVec2DSymmetric},
    units::{Force, ForcePerLength, ForcePerVelocity},
};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, Current, Force>;
pub type NodalStiffnessesSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerLength>;
pub type NodalDampingsSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerVelocity>;
pub type NodalDampingsSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, ForcePerVelocity>;
pub type NodalStiffnessesSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, ForcePerLength>;

pub trait SolidElements {
    type DeformationGradients;
    type DeformationGradientRates;
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients>;
    fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<Self::DeformationGradientRates>;
}
