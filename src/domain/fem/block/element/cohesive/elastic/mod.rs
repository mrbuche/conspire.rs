use crate::constitutive::cohesive::StiffnessCohesive;
use crate::{
    constitutive::cohesive::elastic::Elastic,
    fem::block::element::{
        ElementNodalCoordinates, FiniteElementError,
        cohesive::{CohesiveElement, CohesiveFiniteElement},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
        surface::SurfaceFiniteElement,
    },
    math::TensorList,
    mechanics::TractionList,
};

pub type StiffnessCohesiveList<const N: usize> = TensorList<StiffnessCohesive, N>;

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
        let tractions = Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .map(|(separation, normal)| constitutive_model.traction(separation, normal))
            .collect::<Result<TractionList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(self.nodal_forces_from(tractions))
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        let nodal_mid_surface = Self::nodal_mid_surface(nodal_coordinates);
        let normals = Self::normals(&nodal_mid_surface);
        let normal_gradients = Self::normal_gradients_full(&nodal_mid_surface);
        let stiffnesses = Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .map(|(separation, normal)| constitutive_model.stiffness(separation, normal))
            .collect::<Result<StiffnessCohesiveList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(self.nodal_stiffnesses_from(stiffnesses, normal_gradients))
    }
}
