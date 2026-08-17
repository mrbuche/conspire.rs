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
    math::{Erase, Jacobian, Matrix, Quantity, Scalar, Solution, Tensor, Vector},
    units::{Dimensionless, Energy, UnitDiv},
};
use std::ops::{Div, Mul};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, V, E>
    HyperelasticIVElements<G, V, 3> for Block<C, F, G, M, N, P>
where
    C: HyperelasticIV<V>,
    C::Residual: Erase<Erased = E>,
    F: HyperelasticIVFiniteElement<C, G, M, N, P, V, E>,
    Self: ElasticIVElements<G, V, 3>,
    E: Tensor,
    for<'a> &'a C::Residual: Div<C::TangentVv, Output = V>,
    for<'a> &'a V: Mul<Quantity<Dimensionless>, Output = V> + Mul<Scalar, Output = V>,
    for<'a> &'a Matrix: Mul<&'a V, Output = Vector>,
    V: Erase<Erased = E> + Jacobian + Solution,
    <V as Tensor>::Unit: UnitDiv<<V as Tensor>::Unit, Output = Dimensionless>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        internal_variables: &InternalVariablesField<G, V>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
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
