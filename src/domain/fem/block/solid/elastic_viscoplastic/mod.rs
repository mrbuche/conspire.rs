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
            elastic_viscoplastic::ElasticViscoplasticElements,
        },
    },
    math::{
        Derivative, Differentiate, Quantity, Scalar, Tensor, TensorTupleListVec,
        TensorTupleListVec2D, TensorVector,
        integrate::{
            EmbeddedTableau, EvolvedIncrement, IntegrableField, StateEvolution, Times,
            integrate_rkmk,
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

impl<C, F, const G: usize, const N: usize, const P: usize> Block<C, F, G, 3, N, P>
where
    C: ElasticViscoplastic<Quantity>
        + StateEvolution<
            Time,
            Drive = DeformationGradient,
            Field: IntegrableField<Point = PointStateVariables<Quantity>>,
        >,
    F: ElasticViscoplasticFiniteElement<C, G, 3, N, P, Quantity> + SolidFiniteElement<G, 3, N, P>,
    EvolvedIncrement<C, Time>: Clone + Differentiate<Time>,
    Quantity<Time>: Mul<Scalar, Output = Quantity<Time>>,
    for<'a> &'a Derivative<EvolvedIncrement<C, Time>, Time>:
        Mul<Quantity<Time>, Output = EvolvedIncrement<C, Time>>,
{
    /// Advances every Gauss point's plastic state by one RKMK step over `span`,
    /// with the deformation gradient held frozen at `nodal_coordinates`. `F_p`
    /// stays on the unimodular group (`det = 1`) instead of drifting.
    pub(crate) fn state_variables_rkmk_step<Tab>(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Quantity>,
        span: &[Quantity<Time>],
    ) -> Result<ViscoplasticStateVariables<G, Quantity>, ElementModelError>
    where
        Tab: EmbeddedTableau,
    {
        let model = self.constitutive_model();
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
                        let (_, states): (Times, TensorVector<PointStateVariables<Quantity>>) =
                            integrate_rkmk::<<C as StateEvolution<Time>>::Field, Tab, _, _>(
                                |t, state| model.state_rate(t, &frozen, state),
                                span,
                                point_state.clone(),
                            )
                            .map_err(|error| FiniteElementError::upstream(error, element))?;
                        Ok(states
                            .iter()
                            .last()
                            .expect("the RKMK step yields at least the endpoint")
                            .clone())
                    })
                    .collect::<Result<_, FiniteElementError>>()
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
