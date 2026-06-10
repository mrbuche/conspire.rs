use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, FiniteElementModel, FiniteElementModelError, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::solid::elastic_viscoplastic::ElasticViscoplasticBCs,
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElements,
            elastic::ElasticFiniteElements,
        },
    },
    math::{
        Scalar, Tensor, TensorTuple, TensorVec,
        integrate::{ExplicitDaeFirstOrderRoot, IntegrationError},
        optimize::FirstOrderRootFinding,
    },
    mechanics::Times,
};

pub trait ElasticViscoplasticFiniteElements<S>
where
    Self: SolidFiniteElements,
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

impl<B, S> ElasticViscoplasticFiniteElements<S> for Model<B>
where
    B: ElasticViscoplasticFiniteElements<S>,
{
    fn initial_state(&self) -> S {
        self.blocks.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        self.blocks.nodal_forces(nodal_coordinates, state_variables)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, state_variables)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError> {
        self.blocks
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S> ElasticViscoplasticFiniteElements<S> for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticFiniteElements<S>,
    B2: ElasticFiniteElements,
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

impl<B1, B2, S1, S2> ElasticViscoplasticFiniteElements<TensorTuple<S1, S2>> for Blocks<B1, B2>
where
    B1: ElasticViscoplasticFiniteElements<S1>,
    B2: ElasticViscoplasticFiniteElements<S2>,
    S1: Tensor,
    S2: Tensor,
{
    fn initial_state(&self) -> TensorTuple<S1, S2> {
        (self.0.initial_state(), self.1.initial_state()).into()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, &state_variables.0)?
            + self.1.nodal_forces(nodal_coordinates, &state_variables.1)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, &state_variables.0)?
            + self
                .1
                .nodal_stiffnesses(nodal_coordinates, &state_variables.1)?)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<TensorTuple<S1, S2>, FiniteElementModelError> {
        Ok((
            self.0
                .state_variables_evolution(nodal_coordinates, &state_variables.0)?,
            self.1
                .state_variables_evolution(nodal_coordinates, &state_variables.1)?,
        )
            .into())
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
    B: ElasticViscoplasticFiniteElements<S>,
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
