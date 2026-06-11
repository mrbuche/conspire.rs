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

pub trait ElasticViscoplasticFiniteElements<S, const D: usize>
where
    Self: SolidFiniteElements,
{
    fn initial_state(&self) -> S;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError>;
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError>;
}

impl<B, S, const D: usize> ElasticViscoplasticFiniteElements<S, D> for Model<B, D>
where
    B: ElasticViscoplasticFiniteElements<S, D>,
{
    fn initial_state(&self) -> S {
        self.blocks.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        self.blocks.nodal_forces(nodal_coordinates, state_variables)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, state_variables)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError> {
        self.blocks
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> ElasticViscoplasticFiniteElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticFiniteElements<S, D>,
    B2: ElasticFiniteElements<D>,
{
    fn initial_state(&self) -> S {
        self.0.initial_state()
    }
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, state_variables)?
            + self.1.nodal_forces(nodal_coordinates)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, state_variables)?
            + self.1.nodal_stiffnesses(nodal_coordinates)?)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<S, FiniteElementModelError> {
        self.0
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S1, S2, const D: usize> ElasticViscoplasticFiniteElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: ElasticViscoplasticFiniteElements<S1, D>,
    B2: ElasticViscoplasticFiniteElements<S2, D>,
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
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, &state_variables.0)?
            + self.1.nodal_forces(nodal_coordinates, &state_variables.1)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
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
    B: ElasticViscoplasticFiniteElements<S, D>,
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
