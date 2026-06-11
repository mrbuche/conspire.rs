use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, FiniteElementModel, FiniteElementModelError,
        FiniteElements, Model, NodalCoordinates, NodalCoordinatesHistory,
        block::{
            band_from_neighbors, finalize_node_neighbors,
            solid::elastic_viscoplastic::ElasticViscoplasticBCs,
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic_viscoplastic::ElasticViscoplasticFiniteElements,
            hyperelastic::HyperelasticFiniteElements,
        },
    },
    math::{
        Scalar, Tensor, TensorTuple, TensorVec,
        integrate::{ExplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
};

pub trait HyperelasticViscoplasticFiniteElements<S, const D: usize>
where
    Self: ElasticViscoplasticFiniteElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B, S, const D: usize> HyperelasticViscoplasticFiniteElements<S, D> for Model<B, D>
where
    B: HyperelasticViscoplasticFiniteElements<S, D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> HyperelasticViscoplasticFiniteElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: HyperelasticViscoplasticFiniteElements<S, D>,
    B2: HyperelasticFiniteElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates, state_variables)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}

impl<B1, B2, S1, S2, const D: usize> HyperelasticViscoplasticFiniteElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: HyperelasticViscoplasticFiniteElements<S1, D>,
    B2: HyperelasticViscoplasticFiniteElements<S2, D>,
    S1: Tensor,
    S2: Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<Scalar, FiniteElementModelError> {
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
    B: HyperelasticViscoplasticFiniteElements<S, D>,
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
        let banded = band_from_neighbors(&neighbors, &bcs(time[0]), D);
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
                Some(banded),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
