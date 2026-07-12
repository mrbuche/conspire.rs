use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block, element::solid::elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            parallel_elements,
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic_viscoplastic::ElasticViscoplasticElements,
        },
    },
    math::{
        Scalar, Tensor, TensorTupleListVec, TensorTupleListVec2D, optimize::EqualityConstraint,
    },
    mechanics::DeformationGradientPlastic,
};
use std::array::from_fn;

pub type ViscoplasticStateVariables<const G: usize, Y> =
    TensorTupleListVec<DeformationGradientPlastic, Y, G>;

pub type ViscoplasticStateVariablesHistory<const G: usize, Y> =
    TensorTupleListVec2D<DeformationGradientPlastic, Y, G>;

pub type ElasticViscoplasticBCs = fn(Scalar) -> EqualityConstraint;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    ElasticViscoplasticElements<ViscoplasticStateVariables<G, Y>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticViscoplastic<Y> + Sync,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y> + Sync,
    Y: Tensor + Send + Sync,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<G, Y> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        let states: Vec<_> = state_variables.iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
                states[index],
            )
        }) {
            Ok(forces) => {
                forces.into_iter().flatten().zip(connectivity).for_each(
                    |(element_forces, nodes)| {
                        element_forces
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force)
                    },
                );
                Ok(())
            }
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        let states: Vec<_> = state_variables.iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
                states[index],
            )
        }) {
            Ok(stiffnesses) => {
                stiffnesses
                    .into_iter()
                    .flatten()
                    .zip(connectivity)
                    .for_each(|(element_stiffness, nodes)| {
                        element_stiffness
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(object, &node_a)| {
                                object.into_iter().zip(nodes).for_each(
                                    |(nodal_stiffness, &node_b)| {
                                        nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                                    },
                                )
                            })
                    });
                Ok(())
            }
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<ViscoplasticStateVariables<G, Y>, ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        let states: Vec<_> = state_variables.iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].state_variables_evolution(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
                states[index],
            )
        }) {
            Ok(evolution) => Ok(evolution.into_iter().flatten().collect()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
