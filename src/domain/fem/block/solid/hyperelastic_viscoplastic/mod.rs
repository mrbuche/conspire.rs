use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    fem::{
        NodalCoordinates, NodalCoordinatesHistory,
        block::{
            Block, FiniteElementBlockError, band,
            element::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticFiniteElement,
            solid::{
                NodalForcesSolid, NodalStiffnessesSolid,
                elastic_viscoplastic::{
                    ElasticViscoplasticBCs, ElasticViscoplasticFiniteElementBlock,
                    ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
                },
            },
        },
    },
    math::{
        Scalar, Tensor,
        integrate::{ExplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
};
use std::array::from_fn;

pub trait HyperelasticViscoplasticFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    Y,
> where
    C: HyperelasticViscoplastic<Y>,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Self: ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P, Y>,
    Y: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            ViscoplasticStateVariables<G, Y>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G, Y>,
            NodalCoordinatesHistory,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<
        (
            Times,
            NodalCoordinatesHistory,
            ViscoplasticStateVariablesHistory<G, Y>,
        ),
        IntegrationError,
    >;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    HyperelasticViscoplasticFiniteElementBlock<C, F, G, M, N, P, Y> for Block<C, F, G, M, N, P>
where
    C: HyperelasticViscoplastic<Y>,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Self: ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P, Y>,
    Y: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            ViscoplasticStateVariables<G, Y>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G, Y>,
            NodalCoordinatesHistory,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalCoordinates,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<
        (
            Times,
            NodalCoordinatesHistory,
            ViscoplasticStateVariablesHistory<G, Y>,
        ),
        IntegrationError,
    > {
        let banded = band(
            self.connectivity(),
            &bcs(time[0]),
            self.coordinates().len(),
            3,
        );
        let (time_history, state_variables_history, _, nodal_coordinates_history) = integrator
            .integrate(
                |_: Scalar,
                 state_variables: &ViscoplasticStateVariables<G, Y>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G, Y>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.helmholtz_free_energy(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G, Y>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.nodal_forces(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G, Y>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.nodal_stiffnesses(nodal_coordinates, state_variables)?)
                },
                solver,
                time,
                (
                    self.elements()
                        .iter()
                        .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
                        .collect(),
                    self.coordinates().clone().into(),
                ),
                bcs,
                Some(banded),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
