use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::internal_variables::HyperelasticIV},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError,
        solid::{
            SolidFiniteElement,
            elastic::internal_variables::{ElasticIVFiniteElement, InternalVariables},
        },
    },
    math::{HessianBlock, Jacobian, Scalar, Solution, Tensor},
};

pub trait HyperelasticIVFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    V,
    T1,
    T2,
    T3,
> where
    C: HyperelasticIV<V, T1, T2, T3>,
    Self: ElasticIVFiniteElement<C, G, M, N, P, V, T1, T2, T3>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<Scalar, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize, V, T1, T2, T3>
    HyperelasticIVFiniteElement<C, G, 3, N, P, V, T1, T2, T3> for Element<3, G, N, O>
where
    C: HyperelasticIV<V, T1, T2, T3>,
    Self: ElasticIVFiniteElement<C, G, 3, N, P, V, T1, T2, T3> + SolidFiniteElement<G, 3, N, P>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<Scalar, FiniteElementError> {
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
            .sum::<Result<Scalar, ConstitutiveError>>()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
