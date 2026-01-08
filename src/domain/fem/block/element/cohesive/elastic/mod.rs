use crate::{
    constitutive::cohesive::{elastic::Elastic},
    fem::block::element::{
        ElementNodalCoordinates, FiniteElementError,
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
    ElasticCohesiveElement<C, G, N, P> for CohesiveElement<G, N, O, P>
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
            .map(|separation| {
                constitutive_model.traction(separation)
            })
            .collect::<Result<TractionList<G>, _>>()
        {
            Ok(_) => todo!("Need to do whole -/+ thing too."),
            Err(_)  => todo!(),
        }
        //     Ok(first_piola_kirchhoff_stresses) => Ok(first_piola_kirchhoff_stresses
        //         .iter()
        //         .zip(gradient_vectors.iter().zip(element.integration_weights()))
        //         .map(
        //             |(first_piola_kirchhoff_stress, (gradient_vectors, integration_weight))| {
        //                 gradient_vectors
        //                     .iter()
        //                     .map(|gradient_vector| {
        //                         (first_piola_kirchhoff_stress * gradient_vector) * integration_weight
        //                     })
        //                     .collect()
        //             },
        //         )
        //         .sum()),
        //     Err(error) => Err(FiniteElementError::Upstream(
        //         format!("{error}"),
        //         format!("{element:?}"),
        //     )),
        // }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        todo!()
    }
}
