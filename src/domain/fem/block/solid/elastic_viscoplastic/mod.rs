use crate::mechanics::Times;
use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticStateVariables as PointStateVariables,
        solid::elastic_viscoplastic::ElasticViscoplastic,
    },
    fem::{
        ElementModel, ElementModelError, Model, NodalCoordinates, NodalCoordinatesHistory,
        block::{
            Block,
            element::{
                FiniteElementError,
                solid::{
                    SolidFiniteElement, elastic_viscoplastic::ElasticViscoplasticFiniteElement,
                },
            },
        },
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic_viscoplastic::{
                ElasticViscoplasticElements, ElasticViscoplasticRkmkElements, RootRkmkDae,
            },
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTupleList, TensorTupleListVec,
        TensorTupleListVec2D, TensorVec, TensorVector,
        integrate::{
            ButcherTableau, EmbeddedTableau, EvolvedIncrement, IntegrableField, IntegrationError,
            List, StateEvolution, integrate_rkmk_adaptive, rkmk_dae_step_first_order_root,
            rkmk_step,
        },
        optimize::{EqualityConstraint, FirstOrderRootFinding},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic, DeformationGradientRatePlastic},
    units::Time,
};
use std::array::from_fn;
use std::ops::Mul;

pub type ViscoplasticStateVariables<const G: usize, Y> =
    TensorTupleListVec<DeformationGradientPlastic, Y, G>;

pub type ViscoplasticStateVariablesHistory<const G: usize, Y> =
    TensorTupleListVec2D<DeformationGradientPlastic, Y, G>;

pub type ViscoplasticEvolution<const G: usize, Y> =
    TensorTupleListVec<DeformationGradientRatePlastic, Derivative<Y>, G>;

pub type ViscoplasticEvolutionHistory<const G: usize, Y> =
    TensorTupleListVec2D<DeformationGradientRatePlastic, Derivative<Y>, G>;

pub type ElasticViscoplasticBCs = fn(Quantity<Time>) -> EqualityConstraint;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    ElasticViscoplasticElements<ViscoplasticStateVariables<G, Y>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticViscoplastic<Y>,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Y: Differentiate + Tensor,
{
    fn initial_state(&self) -> ViscoplasticStateVariables<G, Y> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn state_variables_evolution(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<ViscoplasticEvolution<G, Y>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), element_state_variables)| {
                element.state_variables_evolution(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    element_state_variables,
                )
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

/// Advances every Gauss point's plastic state over `[t, t + dt]` with the
/// deformation gradient held frozen at `nodal_coordinates` — a single RKMK step
/// (`tolerances` `None`, one reused stage-slope buffer, no per-Gauss-point
/// allocation) or embedded adaptive substepping (`Some`). `F_p` stays on the
/// unimodular group (`det = 1`) instead of drifting.
impl<C, F, const G: usize, const N: usize, const P: usize, Y> ElasticViscoplasticRkmkElements<Y, 3>
    for Block<C, F, G, 3, N, P>
where
    F: SolidFiniteElement<G, 3, N, P> + ElasticViscoplasticFiniteElement<C, G, 3, N, P, Y>,
    Y: Clone + Differentiate<Time> + Tensor,
    C: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    EvolvedIncrement<C, Time, Y>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time, Y>>,
    ViscoplasticStateVariables<G, Y>: Clone + Differentiate + Tensor,
    ViscoplasticStateVariablesHistory<G, Y>: TensorVec<Item = ViscoplasticStateVariables<G, Y>>,
{
    type State = ViscoplasticStateVariables<G, Y>;
    type History = ViscoplasticStateVariablesHistory<G, Y>;
    fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
        t: Quantity<Time>,
        dt: Quantity<Time>,
        tolerances: Option<(Scalar, Scalar)>,
    ) -> Result<ViscoplasticStateVariables<G, Y>, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        let model = self.constitutive_model();
        let mut scratch: Vec<EvolvedIncrement<C, Time, Y>> = Vec::new();
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), element_state)| {
                let element_coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                element
                    .deformation_gradients(&element_coordinates)
                    .iter()
                    .zip(element_state)
                    .map(|(deformation_gradient, point_state)| {
                        let frozen = deformation_gradient.clone();
                        match tolerances {
                            None => rkmk_step::<<C as StateEvolution<Time, Y>>::Field, Tab, Time>(
                                &mut |t, state| model.state_rate(t, &frozen, state),
                                point_state,
                                t,
                                dt,
                                &mut scratch,
                            ),
                            Some((abs_tol, rel_tol)) => integrate_rkmk_adaptive::<
                                <C as StateEvolution<Time, Y>>::Field,
                                Tab,
                                TensorVector<PointStateVariables<Y>>,
                                Time,
                            >(
                                |t, state| model.state_rate(t, &frozen, state),
                                &[t, t + dt],
                                point_state.clone(),
                                abs_tol,
                                rel_tol,
                            )
                            .map(|(_, history)| {
                                history
                                    .iter()
                                    .last()
                                    .cloned()
                                    .expect("adaptive RKMK window produced no state")
                            }),
                        }
                        .map_err(|error| FiniteElementError::upstream(error, element))
                    })
                    .collect::<Result<_, FiniteElementError>>()
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

/// Flattens the block's grouped per-element state into one Gauss-point list
/// (the [`List`] field's `Point`), in the same element-major order
/// [`ElasticViscoplasticElements`]/[`ElasticViscoplasticRkmkElements`] already
/// iterate — so the round trip through [`unflatten_state`] is exact.
fn flatten_state<const G: usize, Y>(
    state: &ViscoplasticStateVariables<G, Y>,
) -> TensorVector<PointStateVariables<Y>>
where
    Y: Clone + Tensor,
{
    state
        .iter()
        .flat_map(|element_state| element_state.iter().cloned())
        .collect()
}

/// The inverse of [`flatten_state`]: regroups a flat Gauss-point list back
/// into the block's per-element shape, `G` entries per element.
fn unflatten_state<const G: usize, Y>(
    flat: &TensorVector<PointStateVariables<Y>>,
) -> ViscoplasticStateVariables<G, Y>
where
    Y: Clone + Tensor,
{
    flat.as_slice()
        .chunks(G)
        .map(|chunk| {
            chunk
                .iter()
                .cloned()
                .collect::<TensorTupleList<DeformationGradientPlastic, Y, G>>()
        })
        .collect()
}

/// FEM-level RKMK-DAE return map for a single block: nodal equilibrium is
/// resolved from every Gauss point's stage-consistent plastic state at every
/// RK stage abscissa of a load-step window, rather than frozen across it, so
/// the coupling is `Tab`'s own order instead of first order. See
/// [`RootRkmkDae`].
impl<C, F, const G: usize, const N: usize, const P: usize, Y> RootRkmkDae<3, Y>
    for Model<Block<C, F, G, 3, N, P>, 3>
where
    F: SolidFiniteElement<G, 3, N, P> + ElasticViscoplasticFiniteElement<C, G, 3, N, P, Y>,
    Y: Clone + Differentiate<Time> + Tensor,
    C: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Y>>,
        >,
    EvolvedIncrement<C, Time, Y>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time, Y>>,
    Derivative<EvolvedIncrement<C, Time, Y>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time, Y>>,
    TensorVector<PointStateVariables<Y>>: Tensor<Item = PointStateVariables<Y>>,
    TensorVector<EvolvedIncrement<C, Time, Y>>: Tensor<Item = EvolvedIncrement<C, Time, Y>>,
    ViscoplasticStateVariables<G, Y>: Clone,
    ViscoplasticStateVariablesHistory<G, Y>: TensorVec<Item = ViscoplasticStateVariables<G, Y>>,
{
    type History = ViscoplasticStateVariablesHistory<G, Y>;
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
        type Fld<C, Y> = List<<C as StateEvolution<Time, Y>>::Field>;
        let block = self.blocks();
        let model = block.constitutive_model();
        let function = |_: Quantity<Time>,
                        state: &TensorVector<PointStateVariables<Y>>,
                        nodal_coordinates: &NodalCoordinates<3>|
         -> Result<NodalForcesSolid<3>, String> {
            Ok(block.nodal_forces(nodal_coordinates, &unflatten_state::<G, Y>(state))?)
        };
        let jacobian = |_: Quantity<Time>,
                        state: &TensorVector<PointStateVariables<Y>>,
                        nodal_coordinates: &NodalCoordinates<3>|
         -> Result<NodalStiffnessesSolid<3>, String> {
            Ok(block.nodal_stiffnesses(nodal_coordinates, &unflatten_state::<G, Y>(state))?)
        };
        let rate = |t: Quantity<Time>,
                    state: &TensorVector<PointStateVariables<Y>>,
                    nodal_coordinates: &NodalCoordinates<3>|
         -> Result<
            TensorVector<Derivative<EvolvedIncrement<C, Time, Y>, Time>>,
            String,
        > {
            block
                .elements()
                .iter()
                .zip(block.connectivity())
                .enumerate()
                .flat_map(|(e, (element, nodes))| {
                    let element_coordinates =
                        Block::<C, F, G, 3, N, P>::element_coordinates(nodal_coordinates, nodes);
                    element
                        .deformation_gradients(&element_coordinates)
                        .iter()
                        .enumerate()
                        .map(|(g, deformation_gradient)| {
                            model.state_rate(t, deformation_gradient, &state[e * G + g])
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let equality_constraint = bcs;
        let mut state = flatten_state::<G, Y>(&ElasticViscoplasticElements::initial_state(block));
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
        state_variables_history.push(unflatten_state::<G, Y>(&state));
        for step in time.windows(2) {
            let advanced = rkmk_dae_step_first_order_root::<
                Fld<C, Y>,
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
            state_variables_history.push(unflatten_state::<G, Y>(&state));
        }
        Ok((times, nodal_coordinates_history, state_variables_history))
    }
}
