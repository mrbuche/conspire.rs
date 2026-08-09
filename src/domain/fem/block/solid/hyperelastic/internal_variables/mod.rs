use std::ops::{Div, Mul};

use crate::{
    constitutive::solid::hyperelastic::internal_variables::HyperelasticIV,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block, element::solid::hyperelastic::internal_variables::HyperelasticIVFiniteElement,
        },
        solid::{
            elastic::internal_variables::{ElasticIVElements, InternalVariablesField},
            hyperelastic::internal_variables::HyperelasticIVElements,
        },
    },
    math::{Jacobian, Matrix, Scalar, Solution, Vector},
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, V>
    HyperelasticIVElements<G, V, 3> for Block<C, F, G, M, N, P>
where
    C: HyperelasticIV<V>,
    F: HyperelasticIVFiniteElement<C, G, M, N, P, V>,
    Self: ElasticIVElements<G, V, 3>,
    for<'a> &'a V: Div<C::TangentVv, Output = V> + From<&'a V> + Mul<Scalar, Output = V>,
    for<'a> &'a Matrix: Mul<&'a V, Output = Vector>,
    V: Jacobian + Solution,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<Scalar, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(internal_variables)
            .map(|((element, nodes), internal_variables_element)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    internal_variables_element,
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
