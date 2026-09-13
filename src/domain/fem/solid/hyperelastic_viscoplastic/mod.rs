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
            elastic_viscoplastic::{ElasticViscoplasticDaeElements, ElasticViscoplasticElements},
            hyperelastic::HyperelasticElements,
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Tensor, TensorTuple, TensorVec,
        integrate::{
            ButcherTableau, ExplicitDaeSecondOrderMinimize, IntegrableField, IntegrationError,
            rkmk_dae_step_second_order_minimize,
        },
        optimize::SecondOrderOptimization,
    },
    mechanics::Times,
    units::{Energy, Time},
};
use std::ops::Mul;

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

/// The minimize-based sibling of
/// [`crate::domain::fem::solid::elastic_viscoplastic::RootRkmkDae`]: `F` (here,
/// nodal equilibrium) is resolved by potential minimization at every RK stage
/// abscissa from every Gauss point's stage-consistent plastic state, for
/// models whose equilibrium is naturally posed that way rather than as a
/// stress residual. One blanket impl over any [`Model`] whose blocks are
/// [`ElasticViscoplasticDaeElements`] (the topology/flatten/unflatten/rate
/// machinery, shared with the root-finding sibling since the field shape
/// doesn't care how a stage is solved) and [`HyperelasticViscoplasticElements`]
/// (for the potential itself) — a single [`Block`](crate::fem::block::Block),
/// nested [`Blocks`] to any depth, or an [`ElasticViscoplasticAndElastic`]
/// pairing.
pub trait RootRkmkDaeMinimize<const D: usize, Y = Quantity> {
    /// The model's plastic-state history type.
    type History;
    /// Solve under an applied load, resolving nodal equilibrium by potential
    /// minimization at every RK stage of every load-step window while every
    /// Gauss point's plastic state advances on its group.
    fn root_rkmk_dae_minimize<Tab: ButcherTableau>(
        &self,
        solver: impl SecondOrderOptimization<
            Quantity<Energy>,
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, Self::History), IntegrationError>;
}

impl<B, Y> RootRkmkDaeMinimize<3, Y> for Model<B, 3>
where
    B: ElasticViscoplasticDaeElements<Y, 3> + HyperelasticViscoplasticElements<B::State, 3>,
    <B::Field as IntegrableField>::Point: Clone,
    <B::Field as IntegrableField>::Increment: Clone + Differentiate<Time>,
    for<'a> &'a Derivative<<B::Field as IntegrableField>::Increment, Time>:
        Mul<Quantity<Time>, Output = <B::Field as IntegrableField>::Increment>,
    Derivative<<B::Field as IntegrableField>::Increment, Time>:
        Mul<Quantity<Time>, Output = <B::Field as IntegrableField>::Increment>,
    B::State: Clone,
    B::History: TensorVec<Item = B::State>,
{
    type History = B::History;
    #[allow(clippy::type_complexity)]
    fn root_rkmk_dae_minimize<Tab: ButcherTableau>(
        &self,
        solver: impl SecondOrderOptimization<
            Quantity<Energy>,
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinates<3>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<3>, Self::History), IntegrationError> {
        let blocks = self.blocks();
        let mut neighbors = vec![Vec::new(); self.coordinates().len()];
        self.node_neighbors(&mut neighbors);
        finalize_node_neighbors(&mut neighbors);
        let sparse = solver_from_neighbors(&neighbors, &bcs(time[0]), 3, true);
        let function = |_: Quantity<Time>,
                        state: &<B::Field as IntegrableField>::Point,
                        nodal_coordinates: &NodalCoordinates<3>|
         -> Result<Quantity<Energy>, String> {
            Ok(blocks.helmholtz_free_energy(nodal_coordinates, &B::unflatten(state))?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &<B::Field as IntegrableField>::Point,
                        nodal_coordinates: &NodalCoordinates<3>|
         -> Result<NodalForcesSolid<3>, String> {
            Ok(blocks.nodal_forces(nodal_coordinates, &B::unflatten(state))?)
        };
        let hessian = |_: Quantity<Time>,
                       state: &<B::Field as IntegrableField>::Point,
                       nodal_coordinates: &NodalCoordinates<3>|
         -> Result<NodalStiffnessesSolid<3>, String> {
            Ok(blocks.nodal_stiffnesses(nodal_coordinates, &B::unflatten(state))?)
        };
        let rate = |t: Quantity<Time>,
                    state: &<B::Field as IntegrableField>::Point,
                    nodal_coordinates: &NodalCoordinates<3>|
         -> Result<
            Derivative<<B::Field as IntegrableField>::Increment, Time>,
            String,
        > { Ok(blocks.dae_rate(t, nodal_coordinates, state)?) };
        let equality_constraint = bcs;
        let mut state = B::flatten(&ElasticViscoplasticElements::initial_state(blocks));
        let guess: NodalCoordinates<3> = self.coordinates().clone().into();
        let mut nodal_coordinates = solver
            .minimize(
                |x: &NodalCoordinates<3>| function(time[0], &state, x),
                |x: &NodalCoordinates<3>| jacobian(time[0], &state, x),
                |x: &NodalCoordinates<3>| hessian(time[0], &state, x),
                guess,
                equality_constraint(time[0]),
                Some(sparse.clone()),
            )
            .map_err(|error| IntegrationError::from(format!("{error:?}")))?;
        let mut times = Times::new();
        let mut nodal_coordinates_history = NodalCoordinatesHistory::new();
        let mut state_variables_history = Self::History::new();
        let mut scratch = Vec::new();
        let mut carry = None;
        times.push(time[0]);
        nodal_coordinates_history.push(nodal_coordinates.clone());
        state_variables_history.push(B::unflatten(&state));
        for step in time.windows(2) {
            let advanced = rkmk_dae_step_second_order_minimize::<
                B::Field,
                Tab,
                Quantity<Energy>,
                NodalForcesSolid<3>,
                NodalStiffnessesSolid<3>,
                NodalCoordinates<3>,
                Time,
            >(
                &mut |t, state, nodal_coordinates| rate(t, state, nodal_coordinates),
                function,
                jacobian,
                hessian,
                &solver,
                &state,
                &nodal_coordinates,
                step[0],
                step[1] - step[0],
                &mut scratch,
                carry.as_ref(),
                equality_constraint,
                Some(sparse.clone()),
            )
            .map_err(|error| IntegrationError::from(format!("{error:?}")))?;
            state = advanced.0;
            nodal_coordinates = advanced.1;
            carry = advanced.2;
            times.push(step[1]);
            nodal_coordinates_history.push(nodal_coordinates.clone());
            state_variables_history.push(B::unflatten(&state));
        }
        Ok((times, nodal_coordinates_history, state_variables_history))
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
