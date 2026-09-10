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
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTuple, TensorTupleVec,
        TensorVec,
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

/// The RKMK analogue of [`ElasticViscoplasticElements::state_variables_evolution`]:
/// advances a whole block's plastic state one operator-split RKMK step with the
/// deformation gradient frozen at `nodal_coordinates`. Composes over the
/// multi-block wrappers, so [`RkmkRoot`] serves `Blocks` and
/// `ElasticViscoplasticAndElastic` too.
pub trait ElasticViscoplasticRkmkElements<S, const D: usize>
where
    Self: Elements,
    S: Differentiate,
{
    /// One RKMK step for the block's plastic state, `F` frozen.
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        t: Quantity<Time>,
        dt: Quantity<Time>,
    ) -> Result<S, ElementModelError>
    where
        Tab: EmbeddedTableau;
}

impl<B, S, const D: usize> ElasticViscoplasticRkmkElements<S, D> for Model<B, D>
where
    B: ElasticViscoplasticRkmkElements<S, D>,
    S: Differentiate,
{
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        t: Quantity<Time>,
        dt: Quantity<Time>,
    ) -> Result<S, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        self.blocks
            .state_variables_rkmk_step::<Tab>(nodal_coordinates, state_variables, t, dt)
    }
}

impl<B1, B2, S1, S2, const D: usize> ElasticViscoplasticRkmkElements<TensorTuple<S1, S2>, D>
    for Blocks<B1, B2>
where
    B1: ElasticViscoplasticRkmkElements<S1, D>,
    B2: ElasticViscoplasticRkmkElements<S2, D>,
    S1: Differentiate + Tensor,
    S2: Differentiate + Tensor,
    Derivative<S1>: Tensor,
    Derivative<S2>: Tensor,
{
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &TensorTuple<S1, S2>,
        t: Quantity<Time>,
        dt: Quantity<Time>,
    ) -> Result<TensorTuple<S1, S2>, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        Ok((
            self.0.state_variables_rkmk_step::<Tab>(
                nodal_coordinates,
                &state_variables.0,
                t,
                dt,
            )?,
            self.1.state_variables_rkmk_step::<Tab>(
                nodal_coordinates,
                &state_variables.1,
                t,
                dt,
            )?,
        )
            .into())
    }
}

impl<B1, B2, S, const D: usize> ElasticViscoplasticRkmkElements<S, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticRkmkElements<S, D>,
    B2: ElasticElements<D>,
    S: Differentiate,
{
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        t: Quantity<Time>,
        dt: Quantity<Time>,
    ) -> Result<S, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        self.0
            .state_variables_rkmk_step::<Tab>(nodal_coordinates, state_variables, t, dt)
    }
}

/// The shared operator-split (Lie–Trotter) loop behind every [`RkmkRoot`] impl:
/// equilibrium is solved once at the initial time, then each step advances the
/// plastic state one RKMK step with `F` frozen and re-solves equilibrium at the
/// new time with the advanced state held.
fn root_rkmk_operator_split<M, S, H, Tab>(
    model: &M,
    solver: impl FirstOrderRootFinding<
        NodalForcesSolid<3>,
        NodalStiffnessesSolid<3>,
        NodalCoordinates<3>,
    >,
    time: &[Quantity<Time>],
    bcs: ElasticViscoplasticBCs,
) -> Result<(Times, NodalCoordinatesHistory<3>, H), IntegrationError>
where
    M: ElementModel<3> + ElasticViscoplasticElements<S, 3> + ElasticViscoplasticRkmkElements<S, 3>,
    S: Clone + Differentiate + Tensor,
    H: TensorVec<Item = S>,
    Tab: EmbeddedTableau,
{
    let mut state = ElasticViscoplasticElements::initial_state(model);
    let equilibrate = |state: &S,
                       guess: &NodalCoordinates<3>,
                       t: Quantity<Time>|
     -> Result<NodalCoordinates<3>, IntegrationError> {
        solver
            .root(
                |coordinates: &NodalCoordinates<3>| Ok(model.nodal_forces(coordinates, state)?),
                |coordinates: &NodalCoordinates<3>| {
                    Ok(model.nodal_stiffnesses(coordinates, state)?)
                },
                guess.clone(),
                bcs(t),
                None,
            )
            .map_err(|error| IntegrationError::from(format!("{error:?}")))
    };
    let guess: NodalCoordinates<3> = model.coordinates().clone().into();
    let mut nodal_coordinates = equilibrate(&state, &guess, time[0])?;
    let mut times = Times::new();
    let mut nodal_coordinates_history = NodalCoordinatesHistory::new();
    let mut state_variables_history = H::new();
    times.push(time[0]);
    nodal_coordinates_history.push(nodal_coordinates.clone());
    state_variables_history.push(state.clone());
    for step in time.windows(2) {
        state = model
            .state_variables_rkmk_step::<Tab>(
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

/// Interim RKMK return map — an operator-split alternative to
/// [`FirstOrderRoot::root`] that advances every Gauss point's plastic state on
/// its manifold (`F_p` stays unimodular) rather than marching it additively.
///
/// A Lie–Trotter split (see [`root_rkmk_operator_split`]): nodal equilibrium
/// `nodal_forces = λ` is solved once at the initial time, then each load step
/// advances every viscoplastic block's plastic state one RKMK step with the
/// deformation gradient frozen and re-solves equilibrium at the new time with the
/// advanced plastic state held — so every recorded `(t, coordinates, state)` is
/// mutually consistent. Impl'd for a single [`Block`], for two viscoplastic
/// blocks ([`Blocks`]), and for a viscoplastic block paired with a pure-elastic
/// one ([`ElasticViscoplasticAndElastic`]). First order in the coupling; a
/// monolithic version is future work — see the heterogeneous-integration notes.
pub trait RkmkRoot<const D: usize, Y = Quantity>
where
    Y: Differentiate + Tensor,
{
    /// The model's plastic-state history type — a per-Gauss-point list history
    /// for one block, a [`TensorTuple`] of those for [`Blocks`].
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
    BlockStateVariables<G, Y>: Clone + Tensor,
    Self: ElasticViscoplasticElements<BlockStateVariables<G, Y>, 3>
        + ElasticViscoplasticRkmkElements<BlockStateVariables<G, Y>, 3>,
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
        root_rkmk_operator_split::<Self, BlockStateVariables<G, Y>, Self::History, Tab>(
            self, solver, time, bcs,
        )
    }
}

impl<
    C1,
    F1,
    C2,
    F2,
    const G1: usize,
    const N1: usize,
    const P1: usize,
    const G2: usize,
    const N2: usize,
    const P2: usize,
    Y,
> RkmkRoot<3, Y> for Model<Blocks<Block<C1, F1, G1, 3, N1, P1>, Block<C2, F2, G2, 3, N2, P2>>, 3>
where
    Y: Clone + Differentiate<Time> + Tensor,
    C1: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    C2: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    F1: ElasticViscoplasticFiniteElement<C1, G1, 3, N1, P1, Y> + SolidFiniteElement<G1, 3, N1, P1>,
    F2: ElasticViscoplasticFiniteElement<C2, G2, 3, N2, P2, Y> + SolidFiniteElement<G2, 3, N2, P2>,
    EvolvedIncrement<C1, Time, Y>: Clone + Differentiate<Time>,
    EvolvedIncrement<C2, Time, Y>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C1, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C1, Time, Y>>,
    for<'a> &'a Derivative<EvolvedIncrement<C2, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C2, Time, Y>>,
    BlockStateVariables<G1, Y>: Clone + Differentiate + Tensor,
    BlockStateVariables<G2, Y>: Clone + Differentiate + Tensor,
    Derivative<BlockStateVariables<G1, Y>>: Tensor,
    Derivative<BlockStateVariables<G2, Y>>: Tensor,
    TensorTuple<BlockStateVariables<G1, Y>, BlockStateVariables<G2, Y>>:
        Clone + Differentiate + Tensor,
    Self: ElasticViscoplasticElements<
            TensorTuple<BlockStateVariables<G1, Y>, BlockStateVariables<G2, Y>>,
            3,
        > + ElasticViscoplasticRkmkElements<
            TensorTuple<BlockStateVariables<G1, Y>, BlockStateVariables<G2, Y>>,
            3,
        >,
{
    type History = TensorTupleVec<BlockStateVariables<G1, Y>, BlockStateVariables<G2, Y>>;
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
        root_rkmk_operator_split::<
            Self,
            TensorTuple<BlockStateVariables<G1, Y>, BlockStateVariables<G2, Y>>,
            Self::History,
            Tab,
        >(self, solver, time, bcs)
    }
}

impl<
    C1,
    F1,
    C2,
    F2,
    const G1: usize,
    const N1: usize,
    const P1: usize,
    const G2: usize,
    const N2: usize,
    const P2: usize,
    Y,
> RkmkRoot<3, Y>
    for Model<
        ElasticViscoplasticAndElastic<Block<C1, F1, G1, 3, N1, P1>, Block<C2, F2, G2, 3, N2, P2>>,
        3,
    >
where
    Y: Clone + Differentiate<Time> + Tensor,
    C1: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    F1: ElasticViscoplasticFiniteElement<C1, G1, 3, N1, P1, Y> + SolidFiniteElement<G1, 3, N1, P1>,
    Block<C2, F2, G2, 3, N2, P2>: ElasticElements<3>,
    EvolvedIncrement<C1, Time, Y>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C1, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C1, Time, Y>>,
    BlockStateVariables<G1, Y>: Clone + Tensor,
    Self: ElasticViscoplasticElements<BlockStateVariables<G1, Y>, 3>
        + ElasticViscoplasticRkmkElements<BlockStateVariables<G1, Y>, 3>,
{
    type History = BlockStateVariablesHistory<G1, Y>;
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
        root_rkmk_operator_split::<Self, BlockStateVariables<G1, Y>, Self::History, Tab>(
            self, solver, time, bcs,
        )
    }
}
