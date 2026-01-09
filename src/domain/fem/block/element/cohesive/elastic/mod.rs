use crate::{
    constitutive::cohesive::elastic::Elastic,
    fem::block::element::{
        ElementNodalCoordinates, FiniteElement, FiniteElementError,
        cohesive::{CohesiveElement, CohesiveFiniteElement},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
    math::{Rank2, Tensor},
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
        let rotations = Self::rotations(&Self::nodal_mid_surface(nodal_coordinates));
        match Self::separations(nodal_coordinates)
            .into_iter()
            .zip(rotations.iter())
            .map(|(separation, rotation)| constitutive_model.traction(&(rotation * separation)))
            .collect::<Result<TractionList<G>, _>>()
        {
            Ok(tractions) => Ok(tractions
                .into_iter()
                .zip(rotations)
                .zip(
                    Self::signed_shape_functions()
                        .into_iter()
                        .zip(self.integration_weights()),
                )
                .map(
                    |((traction, rotation), (signed_shape_functions, integration_weight))| {
                        let rotated_traction = rotation.transpose() * traction;
                        signed_shape_functions
                            .iter()
                            .map(|signed_shape_function| {
                                &rotated_traction * (signed_shape_function * integration_weight)
                            })
                            .collect()
                    },
                )
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
