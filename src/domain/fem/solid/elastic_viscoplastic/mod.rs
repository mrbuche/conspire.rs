use crate::{
    constitutive::solid::elastic_viscoplastic::ElasticViscoplastic,
    fem::{
        ElasticViscoplasticAndElastic, FiniteElementModel, FiniteElementModelError, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::{
            Block,
            element::solid::elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            solid::elastic_viscoplastic::{
                ElasticViscoplasticBCs, ElasticViscoplasticFiniteElementBlock,
                ViscoplasticStateVariables,
            },
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElementModel,
            elastic::ElasticFiniteElementModel,
        },
    },
    math::{
        Scalar, Tensor, TensorVec,
        integrate::{ExplicitDaeFirstOrderRoot, IntegrationError},
        optimize::FirstOrderRootFinding,
    },
    mechanics::Times,
};

pub trait ElasticViscoplasticFiniteElementModel<S>
where
    Self: SolidFiniteElementModel,
{
    fn initial_state(&self) -> S;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalForcesSolid, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError>;
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    ElasticViscoplasticFiniteElementModel<ViscoplasticStateVariables<G, Y>>
    for Block<C, F, G, M, N, P>
where
    C: ElasticViscoplastic<Y>,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Self: ElasticViscoplasticFiniteElementBlock<C, F, G, M, N, P, Y> + SolidFiniteElementModel,
    Y: Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<G, Y> {
        ElasticViscoplasticFiniteElementBlock::initial_state(self)
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        match ElasticViscoplasticFiniteElementBlock::nodal_forces(
            self,
            nodal_coordinates,
            state_variables,
        ) {
            Ok(nodal_forces) => Ok(nodal_forces),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        match ElasticViscoplasticFiniteElementBlock::nodal_stiffnesses(
            self,
            nodal_coordinates,
            state_variables,
        ) {
            Ok(nodal_stiffnesses) => Ok(nodal_stiffnesses),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<ViscoplasticStateVariables<G, Y>, FiniteElementModelError> {
        match ElasticViscoplasticFiniteElementBlock::state_variables_evolution(
            self,
            nodal_coordinates,
            state_variables,
        ) {
            Ok(state_variables_evolution) => Ok(state_variables_evolution),
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

impl<B1, B2, S> ElasticViscoplasticFiniteElementModel<S> for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticFiniteElementModel<S>,
    B2: ElasticFiniteElementModel,
{
    fn initial_state(&self) -> S {
        self.0.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, state_variables)?
            + self.1.nodal_forces(nodal_coordinates)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, state_variables)?
            + self.1.nodal_stiffnesses(nodal_coordinates)?)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError> {
        self.0
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

pub trait FirstOrderRoot<S, H>
where
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn root(
        &self,
        integrator: impl ExplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            S,
            NodalCoordinates,
            H,
            NodalCoordinatesHistory,
        >,
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory, H), IntegrationError>;
}

impl<B, S, H> FirstOrderRoot<S, H> for Model<B>
where
    B: ElasticViscoplasticFiniteElementModel<S>,
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn root(
        &self,
        integrator: impl ExplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            S,
            NodalCoordinates,
            H,
            NodalCoordinatesHistory,
        >,
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory, H), IntegrationError> {
        let (time_history, state_variables_history, _, nodal_coordinates_history) = integrator
            .integrate(
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates| {
                    Ok(self
                        .blocks
                        .state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates| {
                    Ok(self
                        .blocks
                        .nodal_forces(nodal_coordinates, state_variables)?)
                },
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates| {
                    Ok(self
                        .blocks
                        .nodal_stiffnesses(nodal_coordinates, state_variables)?)
                },
                solver,
                time,
                (
                    self.blocks.initial_state(),
                    self.coordinates().clone().into(),
                ),
                bcs,
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
