#[cfg(test)]
mod test;

use crate::{
    constitutive::cohesive::{damage::Damage, opening_components},
    fem::block::element::{
        ElementNodalCoordinates, FiniteElementError,
        cohesive::{CohesiveElement, CohesiveFiniteElement, elastic::StiffnessCohesiveList},
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
        surface::SurfaceFiniteElement,
    },
    math::{Quantity, Tensor, TensorList, integrate::StepError},
    mechanics::TractionList,
};

/// The greatest opening reached at each integration point of a cohesive element.
pub type DamageState<const G: usize> = TensorList<Quantity, G>;

/// The state of an undamaged cohesive element.
pub fn initial_state<const G: usize>() -> DamageState<G> {
    (0..G).map(|_| Quantity::default()).collect()
}

pub trait DamageCohesiveElement<C, const G: usize, const N: usize, const P: usize>
where
    C: Damage,
    Self: CohesiveFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<DamageState<G>, FiniteElementError>;
    fn step_error(
        &self,
        constitutive_model: &C,
        nodal_coordinates_old: &ElementNodalCoordinates<N>,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_old: &DamageState<G>,
        state: &DamageState<G>,
    ) -> Result<StepError, FiniteElementError>;
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize>
    DamageCohesiveElement<C, G, N, P> for CohesiveElement<G, N, O>
where
    C: Damage,
    Self: CohesiveFiniteElement<G, N, P>,
{
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        let normals = Self::normals(&Self::nodal_mid_surface(nodal_coordinates));
        let tractions = Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .zip(state)
            .map(|((separation, normal), &state)| {
                constitutive_model.traction(separation, normal, state)
            })
            .collect::<Result<TractionList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(self.nodal_forces_from(tractions))
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        let nodal_mid_surface = Self::nodal_mid_surface(nodal_coordinates);
        let normals = Self::normals(&nodal_mid_surface);
        let normal_gradients = Self::normal_gradients_full(&nodal_mid_surface);
        let stiffnesses = Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .zip(state)
            .map(|((separation, normal), &state)| {
                constitutive_model.stiffness(separation, normal, state)
            })
            .collect::<Result<StiffnessCohesiveList<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(self.nodal_stiffnesses_from(stiffnesses, normal_gradients))
    }
    fn updated_state(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state: &DamageState<G>,
    ) -> Result<DamageState<G>, FiniteElementError> {
        let normals = Self::normals(&Self::nodal_mid_surface(nodal_coordinates));
        Self::separations(nodal_coordinates)
            .into_iter()
            .zip(normals)
            .zip(state)
            .map(|((separation, normal), &state)| {
                constitutive_model.state(separation, normal, state)
            })
            .collect::<Result<DamageState<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
    fn step_error(
        &self,
        constitutive_model: &C,
        nodal_coordinates_old: &ElementNodalCoordinates<N>,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        state_old: &DamageState<G>,
        state: &DamageState<G>,
    ) -> Result<StepError, FiniteElementError> {
        let openings = |coordinates: &ElementNodalCoordinates<N>| {
            let normals = Self::normals(&Self::nodal_mid_surface(coordinates));
            Self::separations(coordinates)
                .into_iter()
                .zip(normals)
                .map(|(separation, normal)| opening_components(&separation, &normal))
                .collect::<Vec<_>>()
        };
        openings(nodal_coordinates_old)
            .into_iter()
            .zip(openings(nodal_coordinates))
            .zip(state_old.iter().zip(state))
            .try_fold(
                StepError::default(),
                |worst, ((opening_old, opening), (&state_old, &state))| {
                    constitutive_model
                        .step_error(opening_old, opening, state_old, state)
                        .map(|error| worst.max(error))
                },
            )
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
}
