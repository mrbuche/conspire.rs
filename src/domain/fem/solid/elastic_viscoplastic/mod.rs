use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticStateVariables as PointStateVariables,
        solid::elastic_viscoplastic::ElasticViscoplastic,
    },
    fem::{
        Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Elements, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::{
            Block,
            element::solid::{
                SolidFiniteElement, elastic_viscoplastic::ElasticViscoplasticFiniteElement,
            },
            solid::elastic_viscoplastic::{
                ElasticViscoplasticBCs, ViscoplasticStateVariables as BlockStateVariables,
                ViscoplasticStateVariablesHistory as BlockStateVariablesHistory,
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTuple, TensorVec,
        integrate::{
            EmbeddedTableau, EvolvedIncrement, ExplicitDaeFirstOrderRoot, IntegrableField,
            IntegrationError, StateEvolution,
        },
        optimize::FirstOrderRootFinding,
    },
    mechanics::{DeformationGradient, Times},
    units::Time,
};
use std::ops::Mul;

pub trait ElasticViscoplasticElements<S, const D: usize>
where
    Self: Elements,
    S: Differentiate,
{
    fn initial_state(&self) -> S;
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        let mut nodal_forces = NodalForcesSolid::zero(nodal_coordinates.len());
        self.nodal_forces_into(nodal_coordinates, state_variables, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        self.nodal_stiffnesses_into(nodal_coordinates, state_variables, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Derivative<S>, ElementModelError>;
}

impl<B, S, const D: usize> ElasticViscoplasticElements<S, D> for Model<B, D>
where
    B: ElasticViscoplasticElements<S, D>,
    S: Differentiate,
{
    fn initial_state(&self) -> S {
        self.blocks.initial_state()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_coordinates, state_variables, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_coordinates, state_variables, nodal_stiffnesses)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Derivative<S>, ElementModelError> {
        self.blocks
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S, const D: usize> ElasticViscoplasticElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticElements<S, D>,
    B2: ElasticElements<D>,
    S: Differentiate,
{
    fn initial_state(&self) -> S {
        self.0.initial_state()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_forces_into(nodal_coordinates, state_variables, nodal_forces)?;
        self.1.nodal_forces_into(nodal_coordinates, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_coordinates, state_variables, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_coordinates, nodal_stiffnesses)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
    ) -> Result<Derivative<S>, ElementModelError> {
        self.0
            .state_variables_evolution(nodal_coordinates, state_variables)
    }
}

impl<B1, B2, S1, S2, const D: usize> ElasticViscoplasticElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: ElasticViscoplasticElements<S1, D>,
    B2: ElasticViscoplasticElements<S2, D>,
    S1: Differentiate + Tensor,
    S2: Differentiate + Tensor,
    Derivative<S1>: Tensor,
    Derivative<S2>: Tensor,
{
    fn initial_state(&self) -> TensorTuple<S1, S2> {
        (self.0.initial_state(), self.1.initial_state()).into()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
        nodal_forces: &mut NodalForcesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_forces_into(nodal_coordinates, &state_variables.0, nodal_forces)?;
        self.1
            .nodal_forces_into(nodal_coordinates, &state_variables.1, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<D>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_coordinates, &state_variables.0, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_coordinates, &state_variables.1, nodal_stiffnesses)
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
    ) -> Result<Derivative<TensorTuple<S1, S2>>, ElementModelError> {
        Ok((
            self.0
                .state_variables_evolution(nodal_coordinates, &state_variables.0)?,
            self.1
                .state_variables_evolution(nodal_coordinates, &state_variables.1)?,
        )
            .into())
    }
}

pub trait FirstOrderRoot<S, R, H, const D: usize>
where
    S: Differentiate + Tensor,
    R: TensorVec<Item = Derivative<S>>,
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
            R,
        >,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError>;
}

impl<B, S, R, H, const D: usize> FirstOrderRoot<S, R, H, D> for Model<B, D>
where
    B: ElasticViscoplasticElements<S, D>,
    S: Differentiate + Tensor,
    R: TensorVec<Item = Derivative<S>>,
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
            R,
        >,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, H), IntegrationError> {
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
            )?;
        Ok((
            time_history,
            nodal_coordinates_history,
            state_variables_history,
        ))
    }
}

/// Interim RKMK return map for a single viscoplastic block — an operator-split
/// alternative to [`FirstOrderRoot::root`] that advances every Gauss point's
/// plastic state on its manifold (`F_p` stays unimodular) rather than marching
/// it additively.
///
/// A Lie–Trotter split: nodal equilibrium `nodal_forces = λ` is solved once at
/// the initial time, then each load step takes one
/// [`Block::state_variables_rkmk_step`](crate::fem::block::Block) with the
/// deformation gradient frozen and re-solves equilibrium at the new time with the
/// advanced plastic state held — so every recorded `(t, coordinates, state)` is
/// mutually consistent. First order in the coupling; a monolithic version is
/// future work — see the heterogeneous-integration notes.
pub trait RkmkRoot<const D: usize, Y = Quantity>
where
    Y: Differentiate + Tensor,
{
    /// The block's per-Gauss-point plastic-state history type.
    type History;
    /// Solve under an applied load, advancing the plastic state with a
    /// `Tab`-tableau RKMK step at every Gauss point.
    fn root_rkmk<Tab>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, Self::History), IntegrationError>
    where
        Tab: EmbeddedTableau;
}

impl<C, F, const G: usize, const N: usize, const P: usize, Y> RkmkRoot<3, Y>
    for Model<Block<C, F, G, 3, N, P>, 3>
where
    Y: Clone + Differentiate<Time> + Tensor,
    C: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    F: ElasticViscoplasticFiniteElement<C, G, 3, N, P, Y> + SolidFiniteElement<G, 3, N, P>,
    EvolvedIncrement<C, Time, Y>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time, Y>>,
    Model<Block<C, F, G, 3, N, P>, 3>: ElasticViscoplasticElements<BlockStateVariables<G, Y>, 3>,
{
    type History = BlockStateVariablesHistory<G, Y>;
    fn root_rkmk<Tab>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinates<3>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<3>, Self::History), IntegrationError>
    where
        Tab: EmbeddedTableau,
    {
        let mut state = self.blocks.initial_state();
        let equilibrate = |state: &BlockStateVariables<G, Y>,
                           guess: &NodalCoordinates<3>,
                           t: Quantity<Time>|
         -> Result<NodalCoordinates<3>, IntegrationError> {
            solver
                .root(
                    |coordinates: &NodalCoordinates<3>| Ok(self.nodal_forces(coordinates, state)?),
                    |coordinates: &NodalCoordinates<3>| {
                        Ok(self.nodal_stiffnesses(coordinates, state)?)
                    },
                    guess.clone(),
                    bcs(t),
                    None,
                )
                .map_err(|error| IntegrationError::from(format!("{error:?}")))
        };
        let guess: NodalCoordinates<3> = self.coordinates().clone().into();
        let mut nodal_coordinates = equilibrate(&state, &guess, time[0])?;
        let mut times = Times::new();
        let mut nodal_coordinates_history = NodalCoordinatesHistory::new();
        let mut state_variables_history = Self::History::new();
        times.push(time[0]);
        nodal_coordinates_history.push(nodal_coordinates.clone());
        state_variables_history.push(state.clone());
        for step in time.windows(2) {
            state = self
                .blocks
                .state_variables_rkmk_step::<Tab, Y>(
                    &nodal_coordinates,
                    &state,
                    step[0],
                    step[1] - step[0],
                )
                .map_err(|error| IntegrationError::from(format!("{error:?}")))?;
            nodal_coordinates = equilibrate(&state, &nodal_coordinates, step[1])?;
            times.push(step[1]);
            nodal_coordinates_history.push(nodal_coordinates.clone());
            state_variables_history.push(state.clone());
        }
        Ok((times, nodal_coordinates_history, state_variables_history))
    }
}
