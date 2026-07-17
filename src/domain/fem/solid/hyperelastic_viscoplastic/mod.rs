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
        Scalar, Tensor, TensorTuple, TensorVec,
        integrate::{ExplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
};

pub trait HyperelasticViscoplasticElements<S, const D: usize>
where
    Self: ElasticViscoplasticElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, ElementModelError>;
}

impl<B, S, const D: usize> HyperelasticViscoplasticElements<S, D> for Model<B, D>
where
    B: HyperelasticViscoplasticElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, ElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> HyperelasticViscoplasticElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: HyperelasticViscoplasticElements<S, D>,
    B2: HyperelasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, ElementModelError> {
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
    S1: Tensor,
    S2: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<Scalar, ElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates, &state_variables.0)?
            + self
                .1
                .helmholtz_free_energy(nodal_coordinates, &state_variables.1)?)
    }
}

pub trait SecondOrderMinimize<S, H, const D: usize>
where
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError>;
}

impl<B, S, H, const D: usize> SecondOrderMinimize<S, H, D> for Model<B, D>
where
    B: HyperelasticViscoplasticElements<S, D>,
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            S,
            NodalCoordinates<D>,
            H,
            NodalCoordinatesHistory<D>,
        >,
        solver: impl SecondOrderOptimization<
            Scalar,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Scalar],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &bcs(time[0]), D, true);
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
                        .helmholtz_free_energy(nodal_coordinates, state_variables)?)
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
                Some(sparse),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
