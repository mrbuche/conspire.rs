use crate::{
    fem::{
        ElasticViscoplasticAndElastic, FiniteElementModel, FiniteElementModelError, FiniteElements,
        Model, NodalCoordinates, NodalCoordinatesHistory,
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
        Scalar, Tensor, TensorVec,
        integrate::{ExplicitDaeSecondOrderMinimize, IntegrationError},
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
};

pub trait HyperelasticViscoplasticFiniteElements<S>
where
    Self: ElasticViscoplasticFiniteElements<S>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError>;
}

impl<B, S> HyperelasticViscoplasticFiniteElements<S> for Model<B>
where
    B: HyperelasticViscoplasticFiniteElements<S>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError> {
        self.blocks
            .helmholtz_free_energy(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S> HyperelasticViscoplasticFiniteElements<S> for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: HyperelasticViscoplasticFiniteElements<S>,
    B2: HyperelasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
        state_variables: &S,
    ) -> Result<Scalar, FiniteElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates, state_variables)?
            + self.1.helmholtz_free_energy(nodal_coordinates)?)
    }
}

pub trait SecondOrderMinimize<S, H>
where
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            S,
            NodalCoordinates,
            H,
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
    ) -> Result<(Times, NodalCoordinatesHistory, H), IntegrationError>;
}

impl<B, S, H> SecondOrderMinimize<S, H> for Model<B>
where
    B: HyperelasticViscoplasticFiniteElements<S>,
    S: Tensor,
    H: TensorVec<Item = S>,
{
    fn minimize(
        &self,
        integrator: impl ExplicitDaeSecondOrderMinimize<
            Scalar,
            NodalForcesSolid,
            NodalStiffnessesSolid,
            S,
            NodalCoordinates,
            H,
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
    ) -> Result<(Times, NodalCoordinatesHistory, H), IntegrationError> {
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let banded = band_from_neighbors(&neighbors, &bcs(time[0]), 3);
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
                        .helmholtz_free_energy(nodal_coordinates, state_variables)?)
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
                Some(banded),
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}
