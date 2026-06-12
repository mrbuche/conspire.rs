use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Elements, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::solid::elastic_viscoplastic::ElasticViscoplasticBCs,
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
    math::{
        Scalar, Tensor, TensorTuple, TensorVec,
        integrate::{ExplicitDaeFirstOrderRoot, IntegrationError},
        optimize::FirstOrderRootFinding,
    },
    mechanics::Times,
};

pub trait ElasticViscoplasticElements<S, const D: usize>
where
    Self: Elements,
{
    fn initial_state(&self) -> S;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError>;
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, ElementModelError>;
}

impl<B, S, const D: usize> ElasticViscoplasticElements<S, D> for Model<B, D>
where
    B: ElasticViscoplasticElements<S, D>,
{
    fn initial_state(&self) -> S {
        self.blocks.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        self.blocks.nodal_forces(nodal_coordinates, state_variables)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, state_variables)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, ElementModelError> {
        self.blocks
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> ElasticViscoplasticElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticElements<S, D>,
    B2: ElasticElements<D>,
{
    fn initial_state(&self) -> S {
        self.0.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, state_variables)?
            + self.1.nodal_forces(nodal_coordinates)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, state_variables)?
            + self.1.nodal_stiffnesses(nodal_coordinates)?)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, ElementModelError> {
        self.0
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S1, S2, const D: usize> ElasticViscoplasticElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: ElasticViscoplasticElements<S1, D>,
    B2: ElasticViscoplasticElements<S2, D>,
    S1: Tensor,
    S2: Tensor,
{
    fn initial_state(&self) -> TensorTuple<S1, S2> {
        (self.0.initial_state(), self.1.initial_state()).into()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, &state_variables.0)?
            + self.1.nodal_forces(nodal_coordinates, &state_variables.1)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, &state_variables.0)?
            + self
                .1
                .nodal_stiffnesses(nodal_coordinates, &state_variables.1)?)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<TensorTuple<S1, S2>, ElementModelError> {
        Ok((
            self.0
                .state_variables_evolution(nodal_coordinates, &state_variables.0)?,
            self.1
                .state_variables_evolution(nodal_coordinates, &state_variables.1)?,
        )
            .into())
    }
}

pub trait FirstOrderRoot<S, H, const D: usize>
where
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn root(
        &self,
        integrator: impl ExplicitDaeFirstOrderRoot<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
        >,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError>;
}

impl<B, S, H, const D: usize> FirstOrderRoot<S, H, D> for Model<B, D>
where
    B: ElasticViscoplasticElements<S, D>,
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn root(
        &self,
        integrator: impl ExplicitDaeFirstOrderRoot<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
        >,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError> {
        let (time_history, state_variables_history, _, nodal_coordinates_history) = integrator
            .integrate(
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates<D>| {
                    Ok(self
                        .blocks
                        .state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates<D>| {
                    Ok(self
                        .blocks
                        .nodal_forces(nodal_coordinates, state_variables)?)
                },
                |_: Scalar, state_variables: &S, nodal_coordinates: &NodalCoordinates<D>| {
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
