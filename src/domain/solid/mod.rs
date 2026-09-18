pub(crate) mod elastic;
// Not yet used by cbm alone (only via fem/vem); not dead in the architectural
// sense, so suppress rather than gate out
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod elastic_hyperviscous;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod elastic_viscoplastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperelastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperelastic_viscoplastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod hyperviscoelastic;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod viscoelastic;

use crate::{
    domain::{NodalCoordinates, NodalVelocities},
    math::{Current, TensorRank1Vec, TensorRank2SparseVec2D, TensorRank2SparseVec2DSymmetric},
    units::{Force, ForcePerLength, ForcePerVelocity},
};

pub type NodalForcesSolid<const D: usize> = TensorRank1Vec<D, Current, Force>;
pub type NodalStiffnessesSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerLength>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub type NodalDampingsSolid<const D: usize> =
    TensorRank2SparseVec2D<D, Current, Current, ForcePerVelocity>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub type NodalDampingsSolidSymmetric<const D: usize> =
    TensorRank2SparseVec2DSymmetric<D, Current, Current, ForcePerVelocity>;
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
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
