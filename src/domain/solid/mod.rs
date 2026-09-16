pub(crate) mod elastic;
pub(crate) mod hyperelastic;

use crate::{
    domain::{ElementModelError, NodalCoordinates},
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
    fn deformation_gradients(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Vec<Self::DeformationGradients>;
}

pub(crate) fn accumulate_nodal_forces<'a, T>(
    mut elements_and_nodes: impl Iterator<Item = (Result<T, ElementModelError>, &'a [usize])>,
    nodal_forces: &mut NodalForcesSolid<3>,
) -> Result<(), ElementModelError>
where
    T: IntoIterator<Item = crate::mechanics::Force>,
{
    elements_and_nodes.try_for_each(|(forces, nodes)| {
        forces?
            .into_iter()
            .zip(nodes)
            .for_each(|(force, &node)| nodal_forces[node] += force);
        Ok(())
    })
}

pub(crate) fn accumulate_nodal_stiffnesses<'a, T, R>(
    mut elements_and_nodes: impl Iterator<Item = (Result<T, ElementModelError>, &'a [usize])>,
    nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
) -> Result<(), ElementModelError>
where
    T: IntoIterator<Item = R>,
    R: IntoIterator<Item = crate::mechanics::Stiffness>,
{
    elements_and_nodes.try_for_each(|(stiffnesses, nodes)| {
        stiffnesses?
            .into_iter()
            .zip(nodes)
            .for_each(|(row, &node_a)| {
                row.into_iter()
                    .zip(nodes)
                    .for_each(|(stiffness, &node_b)| nodal_stiffnesses[node_a][node_b] += stiffness)
            });
        Ok(())
    })
}
