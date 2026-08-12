use crate::math::unit::Energy;
use crate::math::unit::{Dimensionless, UnitDiv};
use std::ops::{Div, Mul};

use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::internal_variables::HyperelasticIV},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidFiniteElement,
            elastic::internal_variables::{ElasticIVFiniteElement, InternalVariables},
        },
    },
    math::{Erase, Jacobian, Matrix, Quantity, Scalar, Solution, Tensor, Vector},
};

pub trait HyperelasticIVFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    V,
    E,
> where
    C: HyperelasticIV<V>,
    C::Residual: Erase<Erased = E>,
    Self: ElasticIVFiniteElement<C, G, M, N, P, V, E>,
    E: Tensor,
    for<'a> &'a C::Residual: Div<C::TangentVv, Output = V>,
    for<'a> &'a V: Mul<Quantity<Dimensionless>, Output = V> + Mul<Scalar, Output = V>,
    for<'a> &'a Matrix: Mul<&'a V, Output = Vector>,
    V: Erase<Erased = E> + Jacobian + Solution,
    <V as Tensor>::Unit: UnitDiv<<V as Tensor>::Unit, Output = Dimensionless>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<Quantity<Energy>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize, V, E>
    HyperelasticIVFiniteElement<C, G, 3, N, P, V, E> for Element<3, G, N, O>
where
    C: HyperelasticIV<V>,
    C::Residual: Erase<Erased = E>,
    Self: ElasticIVFiniteElement<C, G, 3, N, P, V, E> + SolidFiniteElement<G, 3, N, P>,
    E: Tensor,
    for<'a> &'a C::Residual: Div<C::TangentVv, Output = V>,
    for<'a> &'a V: Mul<Quantity<Dimensionless>, Output = V> + Mul<Scalar, Output = V>,
    for<'a> &'a Matrix: Mul<&'a V, Output = Vector>,
    V: Erase<Erased = E> + Jacobian + Solution,
    <V as Tensor>::Unit: UnitDiv<<V as Tensor>::Unit, Output = Dimensionless>,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(internal_variables)
            .zip(self.integration_weights())
            .map(
                |((deformation_gradient, internal_variables_point), integration_weight)| {
                    Ok(constitutive_model.helmholtz_free_energy_density(
                        deformation_gradient,
                        internal_variables_point,
                    )? * integration_weight)
                },
            )
            .sum::<Result<Quantity<Energy>, ConstitutiveError>>()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
