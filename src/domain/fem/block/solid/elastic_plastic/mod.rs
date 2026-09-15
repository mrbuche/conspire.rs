use crate::{
    constitutive::solid::elastic_plastic::ElasticPlastic,
    fem::{
        ElementModelError, Elements, NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::elastic_plastic::ElasticPlasticFiniteElement},
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic_plastic::ElasticPlasticElements},
    },
    math::{Matrix, Quantity, Scalar, Tensor, TensorTupleListVec, Vector},
    mechanics::DeformationGradientPlastic,
};
use std::array::from_fn;

/// The rate-independent plastic state at every integration point of every element in a block.
pub type PlasticStateVariablesField<const G: usize> =
    TensorTupleListVec<DeformationGradientPlastic, Quantity, G>;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticPlasticElements<PlasticStateVariablesField<G>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticPlastic,
    F: ElasticPlasticFiniteElement<C, G, M, N, P>,
{
    fn initial_state(&self) -> PlasticStateVariablesField<G> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
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
        state_variables: &PlasticStateVariablesField<G>,
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
    fn updated_state(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
    ) -> Result<PlasticStateVariablesField<G>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.updated_state(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .collect::<Result<_, FiniteElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

/// Aggregation for the monolithic (block) solve, with the plastic multiplier field a
/// free unknown of the outer solver rather than condensed out per quadrature point by
/// [`ElasticPlasticElements`].
pub trait MonolithicElasticPlasticElements<const G: usize, const D: usize>
where
    Self: Elements,
{
    /// The number of scalar local unknowns (one plastic multiplier per integration
    /// point of every element).
    fn num_local(&self) -> usize;
    /// The residual and tangent blocks `(r_u, r_v, K_uu, K_uv, K_vu, K_vv)` of the
    /// monolithic system, with `plastic_multipliers` the current trial `\Delta\gamma`
    /// field (one entry per integration point, in element-then-point order).
    fn monolithic_contributions(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &PlasticStateVariablesField<G>,
        plastic_multipliers: &Vector,
    ) -> Result<(NodalForcesSolid<D>, Vector, Matrix, Matrix, Matrix, Matrix), ElementModelError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    MonolithicElasticPlasticElements<G, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticPlastic,
    F: ElasticPlasticFiniteElement<C, G, M, N, P>,
{
    fn num_local(&self) -> usize {
        self.elements().len() * G
    }
    fn monolithic_contributions(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
        plastic_multipliers: &Vector,
    ) -> Result<(NodalForcesSolid<3>, Vector, Matrix, Matrix, Matrix, Matrix), ElementModelError>
    {
        let num_global = 3 * nodal_coordinates.len();
        let num_local = self.num_local();
        let mut residual_global = NodalForcesSolid::zero(nodal_coordinates.len());
        let mut residual_local = Vector::zero(num_local);
        let mut k_uu = Matrix::zero(num_global, num_global);
        let mut k_uv = Matrix::zero(num_global, num_local);
        let mut k_vu = Matrix::zero(num_local, num_global);
        let mut k_vv = Matrix::zero(num_local, num_local);
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .enumerate()
            .try_for_each(
                |(element_index, ((element, nodes), state_variables_element))| {
                    let offset = element_index * G;
                    let multipliers: [Scalar; G] = from_fn(|g| plastic_multipliers[offset + g]);
                    let (
                        element_residual_global,
                        element_residual_local,
                        element_k_uu,
                        element_k_uv,
                        element_k_vu,
                        element_k_vv,
                    ) = element.monolithic_contributions(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        state_variables_element,
                        &multipliers,
                    )?;
                    element_residual_global
                        .into_iter()
                        .zip(nodes)
                        .for_each(|(nodal_force, &node)| residual_global[node] += nodal_force);
                    (0..G).for_each(|g| residual_local[offset + g] = element_residual_local[g]);
                    element_k_uu
                        .into_iter()
                        .zip(nodes)
                        .for_each(|(row, &node_a)| {
                            row.into_iter().zip(nodes).for_each(|(block, &node_b)| {
                                (0..3).for_each(|i| {
                                    (0..3).for_each(|j| {
                                        k_uu[3 * node_a + i][3 * node_b + j] += block[i][j].value()
                                    })
                                })
                            })
                        });
                    (0..G).for_each(|g| {
                        nodes.iter().enumerate().for_each(|(a, &node_a)| {
                            (0..3).for_each(|i| {
                                k_uv[3 * node_a + i][offset + g] += element_k_uv[g][a][i];
                                k_vu[offset + g][3 * node_a + i] += element_k_vu[g][a][i];
                            })
                        });
                        k_vv[offset + g][offset + g] = element_k_vv[g];
                    });
                    Ok::<(), FiniteElementError>(())
                },
            )
            .map_err(|error| ElementModelError::upstream(error, self))?;
        Ok((residual_global, residual_local, k_uu, k_uv, k_vu, k_vv))
    }
}
