use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticStateVariables as PointStateVariables,
        solid::elastic_viscoplastic::ElasticViscoplastic,
    },
    fem::{
        ElementModelError, NodalCoordinates,
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
                ElasticViscoplasticDaeElements, ElasticViscoplasticElements,
                ElasticViscoplasticRkmkElements,
            },
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTupleList, TensorTupleListVec,
        TensorTupleListVec2D, TensorVec, TensorVector,
        integrate::{
            EmbeddedTableau, EvolvedIncrement, IntegrableField, List, StateEvolution,
            integrate_rkmk_adaptive, rkmk_step,
        },
        optimize::EqualityConstraint,
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
/// [`ElasticViscoplasticElements`] already iterates — so the round trip
/// through [`unflatten_state`] is exact. A free function, not a trait method
/// body, because normalizing the nested [`Tensor::Item`] projections here
/// fails inside the [`ElasticViscoplasticDaeElements`] impl's larger
/// where-clause environment.
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

/// The [`ElasticViscoplasticDaeElements`] machinery for a single block: the
/// whole-mesh field is a per-Gauss-point [`List`] of the constitutive model's
/// own [`StateEvolution`] field, flattened/unflattened in the same
/// element-major order [`ElasticViscoplasticElements`] iterates, and the rate
/// at a stage recomputes every Gauss point's deformation gradient from the
/// stage's nodal coordinates before evaluating [`StateEvolution::state_rate`].
impl<C, F, const G: usize, const N: usize, const P: usize, Y> ElasticViscoplasticDaeElements<Y, 3>
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
    TensorVector<PointStateVariables<Y>>: Tensor<Item = PointStateVariables<Y>>,
    TensorVector<EvolvedIncrement<C, Time, Y>>: Tensor<Item = EvolvedIncrement<C, Time, Y>>,
    ViscoplasticStateVariables<G, Y>: Clone + Differentiate + Tensor,
    ViscoplasticStateVariablesHistory<G, Y>: TensorVec<Item = ViscoplasticStateVariables<G, Y>>,
{
    type Field = List<<C as StateEvolution<Time, Y>>::Field>;
    type State = ViscoplasticStateVariables<G, Y>;
    type History = ViscoplasticStateVariablesHistory<G, Y>;
    fn flatten(state: &ViscoplasticStateVariables<G, Y>) -> TensorVector<PointStateVariables<Y>> {
        flatten_state::<G, Y>(state)
    }
    fn unflatten(flat: &TensorVector<PointStateVariables<Y>>) -> ViscoplasticStateVariables<G, Y> {
        unflatten_state::<G, Y>(flat)
    }
    fn dae_rate(
        &self,
        t: Quantity<Time>,
        nodal_coordinates: &NodalCoordinates<3>,
        flat: &TensorVector<PointStateVariables<Y>>,
    ) -> Result<TensorVector<Derivative<EvolvedIncrement<C, Time, Y>, Time>>, ElementModelError>
    {
        let model = self.constitutive_model();
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .flat_map(|(e, (element, nodes))| {
                let element_coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                element
                    .deformation_gradients(&element_coordinates)
                    .iter()
                    .enumerate()
                    .map(|(g, deformation_gradient)| {
                        model.state_rate(t, deformation_gradient, &flat[e * G + g])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Result<_, String>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
