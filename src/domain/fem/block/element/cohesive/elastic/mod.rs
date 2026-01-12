use crate::{
    constitutive::cohesive::elastic::Elastic,
    fem::block::element::{
        ElementNodalCoordinates, FiniteElement, FiniteElementError,
        cohesive::{CohesiveElement, CohesiveFiniteElement},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
        surface::SurfaceFiniteElement,
    },
    math::Tensor,
    mechanics::{StiffnessList, TractionList},
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
        let normals = Self::normals(&Self::nodal_mid_surface(nodal_coordinates));
        match Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .map(|(separation, normal)| constitutive_model.traction(separation, normal))
            .collect::<Result<TractionList<G>, _>>()
        {
            Ok(tractions) => Ok(tractions
                .into_iter()
                .zip(
                    Self::signed_shape_functions()
                        .into_iter()
                        .zip(self.integration_weights()),
                )
                .map(|(traction, (signed_shape_functions, integration_weight))| {
                    signed_shape_functions
                        .iter()
                        .map(|signed_shape_function| {
                            &traction * (signed_shape_function * integration_weight)
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
        let normals = Self::normals(&Self::nodal_mid_surface(nodal_coordinates));
        match Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .map(|(separation, normal)| constitutive_model.stiffness(separation, normal))
            .collect::<Result<StiffnessList<G>, _>>()
        {
            Ok(stiffnesses) => Ok(stiffnesses
                .into_iter()
                .zip(
                    Self::signed_shape_functions()
                        .into_iter()
                        .zip(self.integration_weights()),
                )
                .map(
                    |(stiffness, (signed_shape_functions, integration_weight))| {
                        signed_shape_functions
                            .iter()
                            .map(|signed_shape_function_a| {
                                signed_shape_functions
                                    .iter()
                                    .map(|signed_shape_function_b| {
                                        &stiffness
                                            * (signed_shape_function_a
                                                * signed_shape_function_b
                                                * integration_weight)
                                    })
                                    .collect() // need to add part with normal gradients
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
}
