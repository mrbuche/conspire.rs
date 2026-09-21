use crate::{
    constitutive::{
        ConstitutiveError,
        solid::elastic_plastic::{ElasticPlastic, coupled},
    },
    domain::solid::elastic_plastic::{ElasticPlasticElements, MonolithicSystem},
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{
                FiniteElementError,
                solid::elastic_plastic::{
                    ElasticPlasticFiniteElement, MonolithicElasticPlasticFiniteElement,
                },
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{Tensor, Vector, sparse::CscMatrix},
};
use std::array::from_fn;

pub use crate::domain::block::solid::plastic::PlasticStateVariablesField;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    ElasticPlasticElements<PlasticStateVariablesField<G>, 3> for Block<C, F, G, M, N, P>
where
    C: ElasticPlastic,
    F: ElasticPlasticFiniteElement<C, G, M, N, P> + MonolithicElasticPlasticFiniteElement<C, G, N>,
{
    fn initial_state(&self) -> PlasticStateVariablesField<G> {
        self.elements()
            .iter()
            .map(|_| from_fn(|_| self.constitutive_model().initial_state()).into())
            .collect()
    }
    fn nodal_forces_and_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
        nodal_forces: &mut NodalForcesSolid<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .try_for_each(|((element, nodes), state_variables_element)| {
                let (forces, stiffnesses) = element.nodal_forces_and_stiffnesses(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )?;
                forces
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                stiffnesses
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
    fn monolithic_system(&self, num_nodes: usize) -> MonolithicSystem {
        let size = coupled::SIZE * G;
        let num_global = 3 * num_nodes;
        let num_local = size * self.elements().len();
        let (mut uu, mut uv, mut vu, mut vv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        self.connectivity()
            .iter()
            .enumerate()
            .for_each(|(element, nodes)| {
                let base = size * element;
                let dofs: Vec<usize> = nodes
                    .iter()
                    .flat_map(|&node| (0..3).map(move |i| 3 * node + i))
                    .collect();
                dofs.iter().for_each(|&row| {
                    dofs.iter().for_each(|&column| uu.push((row, column)));
                    (0..size).for_each(|local| {
                        uv.push((row, base + local));
                        vu.push((base + local, row))
                    })
                });
                (0..G).for_each(|g| {
                    (0..coupled::SIZE).for_each(|l| {
                        (0..coupled::SIZE).for_each(|m| {
                            let offset = base + coupled::SIZE * g;
                            vv.push((offset + l, offset + m))
                        })
                    })
                })
            });
        let finish = |mut pattern: Vec<(usize, usize)>, height, width| {
            pattern.sort_unstable();
            pattern.dedup();
            CscMatrix::from_pattern(height, width, pattern)
        };
        MonolithicSystem {
            residual_global: Vector::zero(num_global),
            residual_local: Vector::zero(num_local),
            tangent_uu: finish(uu, num_global, num_global),
            tangent_uv: finish(uv, num_global, num_local),
            tangent_vu: finish(vu, num_local, num_global),
            tangent_vv: finish(vv, num_local, num_local).with_block_size(coupled::SIZE),
        }
    }
    fn monolithic_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<G>,
        local: &Vector,
        system: &mut MonolithicSystem,
    ) -> Result<(), ElementModelError> {
        let size = coupled::SIZE * G;
        system.clear();
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .enumerate()
            .try_for_each(|(index, ((element, nodes), state_variables_element))| {
                let base = size * index;
                let contributions = element.monolithic(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                    &local.as_slice()[base..base + size],
                )?;
                let dofs: Vec<usize> = nodes
                    .iter()
                    .flat_map(|&node| (0..3).map(move |i| 3 * node + i))
                    .collect();
                let num_u = dofs.len();
                dofs.iter().enumerate().for_each(|(row, &dof_row)| {
                    system.residual_global[dof_row] += contributions.residual_global[row];
                    dofs.iter().enumerate().for_each(|(column, &dof_column)| {
                        system.tangent_uu.accumulate(
                            dof_row,
                            dof_column,
                            contributions.tangent_uu[row * num_u + column],
                        )
                    });
                    (0..size).for_each(|column| {
                        system.tangent_uv.accumulate(
                            dof_row,
                            base + column,
                            contributions.tangent_uv[row * size + column],
                        );
                        system.tangent_vu.accumulate(
                            base + column,
                            dof_row,
                            contributions.tangent_vu[column * num_u + row],
                        )
                    })
                });
                (0..size).for_each(|row| {
                    system.residual_local[base + row] = contributions.residual_local[row]
                });
                (0..G).for_each(|g| {
                    (0..coupled::SIZE).for_each(|l| {
                        (0..coupled::SIZE).for_each(|m| {
                            let offset = base + coupled::SIZE * g;
                            system.tangent_vv.accumulate(
                                offset + l,
                                offset + m,
                                contributions.tangent_vv
                                    [(g * coupled::SIZE + l) * coupled::SIZE + m],
                            )
                        })
                    })
                });
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn monolithic_state(
        &self,
        state_variables: &PlasticStateVariablesField<G>,
        local: &Vector,
    ) -> Result<PlasticStateVariablesField<G>, ElementModelError> {
        let size = coupled::SIZE * G;
        state_variables
            .iter()
            .enumerate()
            .map(|(index, state_variables_element)| {
                let states = (0..G)
                    .map(|g| {
                        let offset = size * index + coupled::SIZE * g;
                        coupled::monolithic_state(
                            self.constitutive_model(),
                            &state_variables_element[g],
                            &Vector::from(
                                local.as_slice()[offset..offset + coupled::SIZE].to_vec(),
                            ),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, ConstitutiveError>(from_fn::<_, G, _>(|g| states[g].clone()).into())
            })
            .collect::<Result<_, _>>()
            .map_err(|error| {
                ElementModelError::upstream(FiniteElementError::upstream(error, self), self)
            })
    }
}
