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
            elastic_viscoplastic::{ElasticViscoplasticElements, ElasticViscoplasticRkmkElements},
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTupleListVec,
        TensorTupleListVec2D, TensorVec, TensorVector,
        integrate::{
            EmbeddedTableau, EvolvedIncrement, IntegrableField, StateEvolution,
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
