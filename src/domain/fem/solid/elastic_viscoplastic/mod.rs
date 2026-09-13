use crate::{
    fem::{
        Blocks, ElasticViscoplasticAndElastic, ElementModel, ElementModelError, Elements, Model,
        NodalCoordinates, NodalCoordinatesHistory,
        block::solid::elastic_viscoplastic::ElasticViscoplasticBCs,
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTuple, TensorTupleVec,
        TensorVec,
        integrate::{
            ButcherTableau, EmbeddedTableau, ExplicitDaeFirstOrderRoot, IntegrableField,
            IntegrationError, Product, rkmk_dae_step_first_order_root,
        },
        optimize::FirstOrderRootFinding,
    },
    mechanics::Times,
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
/// advances a whole model's plastic state one operator-split RKMK step with the
/// deformation gradient frozen at `nodal_coordinates`. Composes over the
/// multi-block wrappers (recursively over nested [`Blocks`]), so [`RkmkRoot`]
/// serves any block topology.
pub trait ElasticViscoplasticRkmkElements<Y, const D: usize>
where
    Self: Elements,
{
    /// The composite plastic state — a per-Gauss-point list for one block, a
    /// [`TensorTuple`] of those for [`Blocks`].
    type State: Clone + Differentiate + Tensor;
    /// Time history of [`Self::State`].
    type History: TensorVec<Item = Self::State>;
    /// Advance the plastic state over `[t, t + dt]` with `F` frozen: one fixed
    /// RKMK step when `tolerances` is `None`, or embedded (`Tab::D`) adaptive
    /// substepping to meet `(abs_tol, rel_tol)` when `Some`.
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &Self::State,
        t: Quantity<Time>,
        dt: Quantity<Time>,
        tolerances: Option<(Scalar, Scalar)>,
    ) -> Result<Self::State, ElementModelError>
    where
        Tab: EmbeddedTableau;
}

impl<B, Y, const D: usize> ElasticViscoplasticRkmkElements<Y, D> for Model<B, D>
where
    B: ElasticViscoplasticRkmkElements<Y, D>,
{
    type State = B::State;
    type History = B::History;
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &Self::State,
        t: Quantity<Time>,
        dt: Quantity<Time>,
        tolerances: Option<(Scalar, Scalar)>,
    ) -> Result<Self::State, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        self.blocks.state_variables_rkmk_step::<Tab>(
            nodal_coordinates,
            state_variables,
            t,
            dt,
            tolerances,
        )
    }
}

impl<B1, B2, Y, const D: usize> ElasticViscoplasticRkmkElements<Y, D> for Blocks<B1, B2>
where
    B1: ElasticViscoplasticRkmkElements<Y, D>,
    B2: ElasticViscoplasticRkmkElements<Y, D>,
    Derivative<<B1 as ElasticViscoplasticRkmkElements<Y, D>>::State>: Tensor,
    Derivative<<B2 as ElasticViscoplasticRkmkElements<Y, D>>::State>: Tensor,
    TensorTuple<
        <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State,
        <B2 as ElasticViscoplasticRkmkElements<Y, D>>::State,
    >: Clone + Differentiate + Tensor,
    TensorTupleVec<
        <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State,
        <B2 as ElasticViscoplasticRkmkElements<Y, D>>::State,
    >: TensorVec<
        Item = TensorTuple<
            <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State,
            <B2 as ElasticViscoplasticRkmkElements<Y, D>>::State,
        >,
    >,
{
    type State = TensorTuple<
        <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State,
        <B2 as ElasticViscoplasticRkmkElements<Y, D>>::State,
    >;
    type History = TensorTupleVec<
        <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State,
        <B2 as ElasticViscoplasticRkmkElements<Y, D>>::State,
    >;
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &Self::State,
        t: Quantity<Time>,
        dt: Quantity<Time>,
        tolerances: Option<(Scalar, Scalar)>,
    ) -> Result<Self::State, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        Ok((
            self.0.state_variables_rkmk_step::<Tab>(
                nodal_coordinates,
                &state_variables.0,
                t,
                dt,
                tolerances,
            )?,
            self.1.state_variables_rkmk_step::<Tab>(
                nodal_coordinates,
                &state_variables.1,
                t,
                dt,
                tolerances,
            )?,
        )
            .into())
    }
}

impl<B1, B2, Y, const D: usize> ElasticViscoplasticRkmkElements<Y, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticRkmkElements<Y, D>,
    B2: ElasticElements<D>,
{
    type State = <B1 as ElasticViscoplasticRkmkElements<Y, D>>::State;
    type History = <B1 as ElasticViscoplasticRkmkElements<Y, D>>::History;
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &Self::State,
        t: Quantity<Time>,
        dt: Quantity<Time>,
        tolerances: Option<(Scalar, Scalar)>,
    ) -> Result<Self::State, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        self.0.state_variables_rkmk_step::<Tab>(
            nodal_coordinates,
            state_variables,
            t,
            dt,
            tolerances,
        )
    }
}

/// The shared operator-split (Lie–Trotter) loop behind every [`RkmkRoot`] impl:
/// equilibrium is solved once at the initial time, then each step advances the
/// plastic state with `F` frozen — one fixed RKMK step when `tolerances` is
/// `None`, adaptive substepping to `(abs_tol, rel_tol)` when `Some` — and
/// re-solves equilibrium at the new time with the advanced state held.
#[allow(clippy::type_complexity)]
fn root_rkmk_operator_split<M, Y, Tab>(
    model: &M,
    solver: impl FirstOrderRootFinding<
        NodalForcesSolid<3>,
        NodalStiffnessesSolid<3>,
        NodalCoordinates<3>,
    >,
    time: &[Quantity<Time>],
    bcs: ElasticViscoplasticBCs,
    tolerances: Option<(Scalar, Scalar)>,
) -> Result<
    (
        Times,
        NodalCoordinatesHistory<3>,
        <M as ElasticViscoplasticRkmkElements<Y, 3>>::History,
    ),
    IntegrationError,
>
where
    M: ElementModel<3>
        + ElasticViscoplasticRkmkElements<Y, 3>
        + ElasticViscoplasticElements<<M as ElasticViscoplasticRkmkElements<Y, 3>>::State, 3>,
    Tab: EmbeddedTableau,
{
    let mut state: <M as ElasticViscoplasticRkmkElements<Y, 3>>::State =
        ElasticViscoplasticElements::initial_state(model);
    let equilibrate = |state: &<M as ElasticViscoplasticRkmkElements<Y, 3>>::State,
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
    let mut state_variables_history = <M as ElasticViscoplasticRkmkElements<Y, 3>>::History::new();
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
                tolerances,
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
/// mutually consistent. One blanket impl over any [`Model`] whose blocks are
/// [`ElasticViscoplasticRkmkElements`] — a single [`Block`], nested [`Blocks`] to
/// any depth, or an [`ElasticViscoplasticAndElastic`] pairing. First order in the
/// coupling; a monolithic version is future work — see the
/// heterogeneous-integration notes.
pub trait RkmkRoot<const D: usize, Y = Quantity> {
    /// The model's plastic-state history type — a per-Gauss-point list history
    /// for one block, a [`TensorTuple`] of those for [`Blocks`].
    type History;
    /// Solve under an applied load, advancing every Gauss point's plastic state
    /// one fixed `Tab`-tableau RKMK step per load-step window.
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
    /// As [`Self::root_rkmk`], but every Gauss point's plastic state substeps
    /// within each load-step window under embedded (`Tab::D`) error control to
    /// meet `abs_tol` / `rel_tol`. The equilibrium solve stays once per window,
    /// so the split is still first order in the coupling — this only tightens
    /// the plastic-flow integration for a given load-step grid.
    fn root_rkmk_adaptive<Tab>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
        abs_tol: Scalar,
        rel_tol: Scalar,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, Self::History), IntegrationError>
    where
        Tab: EmbeddedTableau;
}

impl<B, Y> RkmkRoot<3, Y> for Model<B, 3>
where
    B: ElasticViscoplasticRkmkElements<Y, 3>,
    Model<B, 3>: ElementModel<3>
        + ElasticViscoplasticElements<
            <Model<B, 3> as ElasticViscoplasticRkmkElements<Y, 3>>::State,
            3,
        >,
{
    type History = <Model<B, 3> as ElasticViscoplasticRkmkElements<Y, 3>>::History;
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
        root_rkmk_operator_split::<Model<B, 3>, Y, Tab>(self, solver, time, bcs, None)
    }
    fn root_rkmk_adaptive<Tab>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinates<3>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
        abs_tol: Scalar,
        rel_tol: Scalar,
    ) -> Result<(Times, NodalCoordinatesHistory<3>, Self::History), IntegrationError>
    where
        Tab: EmbeddedTableau,
    {
        root_rkmk_operator_split::<Model<B, 3>, Y, Tab>(
            self,
            solver,
            time,
            bcs,
            Some((abs_tol, rel_tol)),
        )
    }
}

/// The RKMK-DAE analogue of [`RkmkRoot`], mirroring the constitutive-level
/// `RootRkmkDae`: `F` (here, nodal equilibrium) is resolved at every RK stage
/// abscissa from every Gauss point's stage-consistent plastic state, rather
/// than frozen across the load-step window — so the coupling is the tableau's
/// own order instead of first order, at the cost of `Tab::STAGES` equilibrium
/// solves per window instead of one. One blanket impl over any [`Model`] whose
/// blocks are [`ElasticViscoplasticDaeElements`] — a single [`Block`], nested
/// [`Blocks`] to any depth, or an [`ElasticViscoplasticAndElastic`] pairing.
pub trait RootRkmkDae<const D: usize, Y = Quantity> {
    /// The model's plastic-state history type.
    type History;
    /// Solve under an applied load, resolving nodal equilibrium at every RK
    /// stage of every load-step window while every Gauss point's plastic
    /// state advances on its group.
    fn root_rkmk_dae<Tab: ButcherTableau>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, Self::History), IntegrationError>;
}

/// Per-topology machinery behind [`RootRkmkDae`]: the whole-mesh
/// [`IntegrableField`] every Gauss point's plastic state lives on (a
/// per-Gauss-point [`List`](crate::math::integrate::List) for one [`Block`],
/// a [`Product`] of those for [`Blocks`]), how to flatten/unflatten between
/// it and the block's native per-element [`State`](Self::State), and how to
/// evaluate every Gauss point's plastic rate at a stage's nodal coordinates.
/// Composes recursively over nested [`Blocks`] and forwards over
/// [`ElasticViscoplasticAndElastic`], the same way
/// [`ElasticViscoplasticElements`] does.
pub trait ElasticViscoplasticDaeElements<Y, const D: usize>
where
    Self: Elements,
{
    /// The whole-mesh field this topology's plastic state lives on.
    type Field: IntegrableField<Increment: Differentiate<Time>>;
    /// The per-element grouped state — a per-Gauss-point list for one block,
    /// a [`TensorTuple`] of those for [`Blocks`].
    type State: Clone + Differentiate + Tensor;
    /// Time history of [`Self::State`].
    type History: TensorVec<Item = Self::State>;
    /// Flattens [`Self::State`] into [`Self::Field`]'s `Point`.
    fn flatten(state: &Self::State) -> <Self::Field as IntegrableField>::Point;
    /// The inverse of [`Self::flatten`].
    fn unflatten(flat: &<Self::Field as IntegrableField>::Point) -> Self::State;
    /// Evaluates every Gauss point's plastic rate at `nodal_coordinates`.
    fn dae_rate(
        &self,
        t: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<D>,
        flat: &<Self::Field as IntegrableField>::Point,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, Time>, ElementModelError>;
}

impl<B1, B2, Y, const D: usize> ElasticViscoplasticDaeElements<Y, D> for Blocks<B1, B2>
where
    B1: ElasticViscoplasticDaeElements<Y, D>,
    B2: ElasticViscoplasticDaeElements<Y, D>,
    TensorTuple<<B1::Field as IntegrableField>::Point, <B2::Field as IntegrableField>::Point>:
        Tensor,
    TensorTuple<
        <B1::Field as IntegrableField>::Increment,
        <B2::Field as IntegrableField>::Increment,
    >: Tensor,
    TensorTuple<B1::State, B2::State>: Clone + Differentiate + Tensor,
    Derivative<B1::State>: Tensor,
    Derivative<B2::State>: Tensor,
    TensorTupleVec<B1::State, B2::State>: TensorVec<Item = TensorTuple<B1::State, B2::State>>,
{
    type Field = Product<B1::Field, B2::Field>;
    type State = TensorTuple<B1::State, B2::State>;
    type History = TensorTupleVec<B1::State, B2::State>;
    fn flatten(state: &Self::State) -> <Self::Field as IntegrableField>::Point {
        TensorTuple(B1::flatten(&state.0), B2::flatten(&state.1))
    }
    fn unflatten(flat: &<Self::Field as IntegrableField>::Point) -> Self::State {
        TensorTuple(B1::unflatten(&flat.0), B2::unflatten(&flat.1))
    }
    fn dae_rate(
        &self,
        t: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<D>,
        flat: &<Self::Field as IntegrableField>::Point,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, Time>, ElementModelError>
    {
        Ok((
            self.0.dae_rate(t, nodal_coordinates, &flat.0)?,
            self.1.dae_rate(t, nodal_coordinates, &flat.1)?,
        )
            .into())
    }
}

impl<B1, B2, Y, const D: usize> ElasticViscoplasticDaeElements<Y, D>
    for ElasticViscoplasticAndElastic<B1, B2>
where
    B1: ElasticViscoplasticDaeElements<Y, D>,
    B2: ElasticElements<D>,
{
    type Field = B1::Field;
    type State = B1::State;
    type History = B1::History;
    fn flatten(state: &Self::State) -> <Self::Field as IntegrableField>::Point {
        B1::flatten(state)
    }
    fn unflatten(flat: &<Self::Field as IntegrableField>::Point) -> Self::State {
        B1::unflatten(flat)
    }
    fn dae_rate(
        &self,
        t: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<D>,
        flat: &<Self::Field as IntegrableField>::Point,
    ) -> Result<Derivative<<Self::Field as IntegrableField>::Increment, Time>, ElementModelError>
    {
        self.0.dae_rate(t, nodal_coordinates, flat)
    }
}

impl<B, Y> RootRkmkDae<3, Y> for Model<B, 3>
where
    B: ElasticViscoplasticDaeElements<Y, 3> + ElasticViscoplasticElements<B::State, 3>,
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
    fn root_rkmk_dae<Tab: ButcherTableau>(
        &self,
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinates<3>,
        >,
        time: &[Quantity<Time>],
        bcs: ElasticViscoplasticBCs,
    ) -> Result<(Times, NodalCoordinatesHistory<3>, Self::History), IntegrationError> {
        let blocks = self.blocks();
        let function = |_: Quantity<Time>,
                        state: &<B::Field as IntegrableField>::Point,
                        nodal_coordinates: &NodalCoordinates<3>|
         -> Result<NodalForcesSolid<3>, String> {
            Ok(blocks.nodal_forces(nodal_coordinates, &B::unflatten(state))?)
        };
        let jacobian = |_: Quantity<Time>,
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
            .root(
                |x: &NodalCoordinates<3>| function(time[0], &state, x),
                |x: &NodalCoordinates<3>| jacobian(time[0], &state, x),
                guess,
                equality_constraint(time[0]),
                None,
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
            let advanced = rkmk_dae_step_first_order_root::<
                B::Field,
                Tab,
                NodalForcesSolid<3>,
                NodalStiffnessesSolid<3>,
                NodalCoordinates<3>,
                Time,
            >(
                &mut |t, state, nodal_coordinates| rate(t, state, nodal_coordinates),
                function,
                jacobian,
                &solver,
                &state,
                &nodal_coordinates,
                step[0],
                step[1] - step[0],
                &mut scratch,
                carry.as_ref(),
                equality_constraint,
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
