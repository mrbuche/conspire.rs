use crate::units::Time;
use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Elements, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::{
            finalize_node_neighbors, solid::elastic_viscoplastic::ElasticViscoplasticBCs,
            solver_from_neighbors,
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic_viscoplastic::ElasticViscoplasticElements, hyperelastic::HyperelasticElements,
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Tensor, TensorTuple, TensorVec,
        integrate::{ExplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
    units::Energy,
};

pub trait HyperelasticViscoplasticElements<S, const D: usize>
where
    Self: ElasticViscoplasticElements<S, D>,
    S: Differentiate,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Quantity<Energy>, ElementModelError>;
}

impl<B, S, const D: usize> HyperelasticViscoplasticElements<S, D> for Model<B, D>
where
    B: HyperelasticViscoplasticElements<S, D>,
    S: Differentiate,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> HyperelasticViscoplasticElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: HyperelasticViscoplasticElements<S, D>,
    B2: HyperelasticElements<D>,
    S: Differentiate,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates, state_variables)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}

impl<B1, B2, S1, S2, const D: usize> HyperelasticViscoplasticElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: HyperelasticViscoplasticElements<S1, D>,
    B2: HyperelasticViscoplasticElements<S2, D>,
    S1: Differentiate + Tensor,
    S2: Differentiate + Tensor,
    Derivative<S1>: Tensor,
    Derivative<S2>: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates, &state_variables.0)?
            + self
                .1
                .helmholtz_free_energy(nodal_coordinates, &state_variables.1)?)
    }
}

pub trait SecondOrderMinimize<S, R, H, const D: usize>
where
    S: Differentiate + Tensor,
    R: TensorVec<Item = Derivative<S>>,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Quantity<Energy>,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
            R,
        >,
        solver: impl SecondOrderOptimization<
            Quantity<Energy>,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError>;
}

impl<B, S, R, H, const D: usize> SecondOrderMinimize<S, R, H, D> for Model<B, D>
where
    B: HyperelasticViscoplasticElements<S, D>,
    S: Differentiate + Tensor,
    R: TensorVec<Item = Derivative<S>>,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Quantity<Energy>,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
            R,
        >,
        solver: impl SecondOrderOptimization<
            Quantity<Energy>,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &bcs(time[0]), D, true);
        let (time_history, state_variables_history, _, nodal_coordinates_history) = integrator
            .integrate(
                |_: Quantity<Time>,
                 state_variables: &S,
                 nodal_coordinates: &NodalCoordinates<D>| {
                    Ok(self
                        .blocks
                        .state_variables_evolution(nodal_coordinates, state_variables)?)
                },
                |_: Quantity<Time>,
                 state_variables: &S,
                 nodal_coordinates: &NodalCoordinates<D>| {
                    Ok(self
                        .blocks
                        .helmholtz_free_energy(nodal_coordinates, state_variables)?)
                },
                |_: Quantity<Time>,
                 state_variables: &S,
                 nodal_coordinates: &NodalCoordinates<D>| {
                    Ok(self
                        .blocks
                        .nodal_forces(nodal_coordinates, state_variables)?)
                },
                |_: Quantity<Time>,
                 state_variables: &S,
                 nodal_coordinates: &NodalCoordinates<D>| {
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
                Some(sparse),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
