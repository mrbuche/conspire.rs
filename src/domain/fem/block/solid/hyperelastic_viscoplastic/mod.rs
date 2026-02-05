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
        Scalar, Tensor, TensorArray,
        integrate::{DaeSolverSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::{DeformationGradientPlastic, Times},
};
use std::array::from_fn;

pub trait HyperelasticViscoplasticFiniteElementBlock<
    C,
    F,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
> where
    C: HyperelasticViscoplastic,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P>,
    Self: ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
    ) -> Result<Scalar, FiniteElementBlockError>;
    fn minimize(
        &self,
        integrator: impl DaeSolverSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            ViscoplasticStateVariables<G>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G>,
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
            ViscoplasticStateVariablesHistory<G>,
        ),
        IntegrationError,
    >;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperelasticViscoplasticFiniteElementBlock<C, F, G, M, N, P> for Block<C, F, G, M, N, P>
where
    C: HyperelasticViscoplastic,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P>,
    Self: ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G>,
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
        integrator: impl DaeSolverSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            ViscoplasticStateVariables<G>,
            NodalCoordinates,
            ViscoplasticStateVariablesHistory<G>,
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
            ViscoplasticStateVariablesHistory<G>,
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
            .integrate_dae(
                |_: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.helmholtz_free_energy(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.nodal_forces(nodal_coordinates, state_variables)?)
                },
                |_t: Scalar,
                 state_variables: &ViscoplasticStateVariables<G>,
                 nodal_coordinates: &NodalCoordinates| {
                    Ok(self.nodal_stiffnesses(nodal_coordinates, state_variables)?)
                },
                solver,
                time,
                (
                    self.elements()
                        .iter()
                        .map(|_| {
                            from_fn(|_| {
                                (
                                    DeformationGradientPlastic::identity(),
                                    self.constitutive_model().initial_yield_stress(),
                                )
                                    .into()
                            })
                            .into()
                        })
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
