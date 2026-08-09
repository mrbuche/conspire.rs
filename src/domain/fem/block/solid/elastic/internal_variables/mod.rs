use std::ops::{Div, Mul};

use crate::{
    constitutive::solid::elastic::internal_variables::ElasticIV,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError, solid::elastic::internal_variables::ElasticIVFiniteElement,
            },
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic::internal_variables::{ElasticIVElements, InternalVariablesField},
        },
    },
    math::{Jacobian, Matrix, Scalar, Solution, Vector, optimize::NewtonRaphson},
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, V>
    ElasticIVElements<G, V, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticIV<V>,
    F: ElasticIVFiniteElement<C, G, M, N, P, V>,
    for<'a> &'a V: Div<C::TangentVv, Output = V> + From<&'a V> + Mul<Scalar, Output = V>,
    for<'a> &'a Matrix: Mul<&'a V, Output = Vector>,
    V: Jacobian + Solution,
{
    fn internal_variables_initial(&self) -> InternalVariablesField<G, V> {
        self.elements()
            .iter()
            .map(|element| element.internal_variables_initial(self.constitutive_model()))
            .collect()
    }
    fn internal_variables_increment(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_decrement: &NodalCoordinates<3>,
        step: Scalar,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .map(|((element, nodes), internal_variables_element)| {
                element.internal_variables_increment(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    internal_variables_element,
                    &Self::element_coordinates(nodal_decrement, nodes),
                    step,
                )
            })
            .collect::<Result<InternalVariablesField<G, V>, _>>()
        {
            Ok(incremented) => Ok(incremented),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn internal_variables_root(
        &self,
        local_solver: &NewtonRaphson,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<InternalVariablesField<G, V>, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .map(|((element, nodes), internal_variables_element)| {
                element.internal_variables_root(
                    local_solver,
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    internal_variables_element,
                )
            })
            .collect::<Result<InternalVariablesField<G, V>, _>>()
        {
            Ok(roots) => Ok(roots),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .try_for_each(|((element, nodes), internal_variables_element)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        internal_variables_element,
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces_eliminated_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .try_for_each(|((element, nodes), internal_variables_element)| {
                element
                    .nodal_forces_eliminated(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        internal_variables_element,
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .try_for_each(|((element, nodes), internal_variables_element)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        internal_variables_element,
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
