use crate::{
    constitutive::cohesive::elastic::Elastic,
    fem::block::element::{
        ElementNodalCoordinates, FiniteElement, FiniteElementError,
        cohesive::{CohesiveElement, CohesiveFiniteElement},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
    math::Tensor,
    mechanics::TractionList,
};

pub trait ElasticCohesiveElement<C, const G: usize, const N: usize, const P: usize>
where
    C: Elastic,
    Self: CohesiveFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    ElasticCohesiveElement<C, G, N, P> for CohesiveElement<G, N, O>
where
    C: Elastic,
    Self: CohesiveFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match Self::separations(nodal_coordinates)
            .iter()
            .map(|separation| constitutive_model.traction(separation))
            .collect::<Result<TractionList<G>, _>>()
        {
            //
            // Need to rotate tractions back to global coordinate system before using below.
            //
            Ok(tractions) => Ok(tractions
                .iter()
                .zip(
                    Self::signed_shape_functions()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(|(traction, (signed_shape_functions, integration_weight))| {
                    signed_shape_functions
                        .iter()
                        .map(|signed_shape_function| {
                            traction * (signed_shape_function * integration_weight)
                        })
                        .collect()
                })
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        todo!()
    }
}
