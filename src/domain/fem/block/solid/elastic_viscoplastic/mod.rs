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
            elastic_viscoplastic::{ElasticViscoplasticDaeElements, ElasticViscoplasticElements},
        },
    },
    math::{
        Derivative, Differentiable, Quantity, Tensor, TensorVec, TensorVector,
        integrate::{EvolvedIncrement, Integrable, List, StateEvolution},
    },
    mechanics::DeformationGradient,
    units::Time,
};
use std::array::from_fn;

pub use crate::domain::{
    block::solid::viscoplastic::{
        ViscoplasticEvolution, ViscoplasticStateVariables, ViscoplasticStateVariablesHistory,
    },
    solid::elastic_viscoplastic::ElasticViscoplasticBCs,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y>
    ElasticViscoplasticElements<ViscoplasticStateVariables<G, Y>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticViscoplastic<Y>,
    F: ElasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Y: Differentiable + Tensor,
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

fn unflatten_state<const G: usize, Y>(
    flat: &TensorVector<PointStateVariables<Y>>,
) -> ViscoplasticStateVariables<G, Y>
where
    Y: Clone + Tensor,
{
    flat.as_slice()
        .chunks(G)
        .map(|chunk| chunk.iter().cloned().collect())
        .collect()
}

impl<C, F, const G: usize, const N: usize, const P: usize, Y> ElasticViscoplasticDaeElements<Y, 3>
    for Block<C, F, G, 3, N, P>
where
    F: SolidFiniteElement<G, 3, N, P> + ElasticViscoplasticFiniteElement<C, G, 3, N, P, Y>,
    Y: Clone + Differentiable<Time> + Tensor,
    C: ElasticViscoplastic<Y>
        + StateEvolution<
            Time,
            Y,
            Drive = DeformationGradient,
            Field: Integrable<Point = PointStateVariables<Y>>,
        >,
    EvolvedIncrement<C, Time, Y>: Clone + Differentiable<Time>,
    TensorVector<PointStateVariables<Y>>: Tensor<Item = PointStateVariables<Y>>,
    TensorVector<EvolvedIncrement<C, Time, Y>>: Tensor<Item = EvolvedIncrement<C, Time, Y>>,
    ViscoplasticStateVariables<G, Y>: Clone + Differentiable + Tensor,
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
