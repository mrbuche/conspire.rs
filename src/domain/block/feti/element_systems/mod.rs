#[cfg(feature = "fem")]
use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::block::{
        Block,
        element::{FiniteElementError, solid::hyperelastic::HyperelasticFiniteElement},
    },
    math::Tensor,
};
use crate::{
    domain::{Blocks, ElementModelError, Model, NodalCoordinates},
    geometry::mesh::Partition,
    math::{Scalar, SquareMatrix, Vector},
};
use std::collections::HashMap;

/// The tangent of a Newton step left unassembled: each element's stiffness and
/// force, in the order the elements were meshed.
///
/// A decomposition assigns whole elements to subdomains, so this is what one
/// needs to build each subdomain's own system.
pub struct ElementSystems {
    pub(crate) number_of_nodes: usize,
    pub(crate) elements: Vec<ElementSystem>,
}

pub(crate) struct ElementSystem {
    pub(crate) nodes: Vec<usize>,
    pub(crate) stiffness: SquareMatrix,
    pub(crate) force: Vector,
}

impl ElementSystem {
    /// Packs an element's forces, `force(a, i)`, and stiffnesses,
    /// `stiffness(a, b, i, j)`, indexed by its nodes and components, into dense
    /// arrays over its degrees of freedom.
    pub(crate) fn pack(
        nodes: Vec<usize>,
        force: impl Fn(usize, usize) -> Scalar,
        stiffness: impl Fn(usize, usize, usize, usize) -> Scalar,
    ) -> Self {
        const D: usize = 3;
        let number_of_nodes = nodes.len();
        let mut packed_stiffness = SquareMatrix::zero(D * number_of_nodes);
        let mut packed_force = Vector::zero(D * number_of_nodes);
        (0..number_of_nodes).for_each(|a| {
            (0..D).for_each(|i| packed_force[D * a + i] = force(a, i));
            (0..number_of_nodes).for_each(|b| {
                (0..D).for_each(|i| {
                    (0..D).for_each(|j| {
                        packed_stiffness[D * a + i][D * b + j] = stiffness(a, b, i, j)
                    })
                })
            })
        });
        Self {
            nodes,
            stiffness: packed_stiffness,
            force: packed_force,
        }
    }
}

/// Elements that can hand out their systems one by one.
///
/// This is what a decomposed solve asks of a model, and a model that cannot
/// answer it, being elastic rather than hyperelastic or having couplings that
/// cross any cut, is refused at compile time.
pub trait DecomposableElements {
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError>;
}

impl<B> DecomposableElements for Model<B, 3>
where
    B: DecomposableElements,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        self.blocks.element_systems(nodal_coordinates)
    }
}

impl<B1, B2> DecomposableElements for Blocks<B1, B2>
where
    B1: DecomposableElements,
    B2: DecomposableElements,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        let mut systems = self.0.element_systems(nodal_coordinates)?;
        systems
            .elements
            .extend(self.1.element_systems(nodal_coordinates)?.elements);
        Ok(systems)
    }
}

#[cfg(feature = "fem")]
impl<C, F, const G: usize, const N: usize, const P: usize> DecomposableElements
    for Block<C, F, G, 3, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, 3, N, P>,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        let elements = self
            .connectivity()
            .iter()
            .zip(self.elements())
            .map(|(nodes, element)| {
                let coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                let forces = element.nodal_forces(self.constitutive_model(), &coordinates)?;
                let stiffnesses =
                    element.nodal_stiffnesses(self.constitutive_model(), &coordinates)?;
                Ok::<_, FiniteElementError>(ElementSystem::pack(
                    nodes.to_vec(),
                    |a, i| forces[a][i].value(),
                    |a, b, i, j| stiffnesses[a][b][i][j].value(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ElementModelError::upstream(error, self))?;
        Ok(ElementSystems {
            number_of_nodes: nodal_coordinates.len(),
            elements,
        })
    }
}

impl ElementSystems {
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.number_of_nodes
    }
    /// Each subdomain's own stiffness and force: the sum of its elements'.
    #[allow(clippy::type_complexity)]
    pub(crate) fn subdomains(
        &self,
        partition: &Partition,
    ) -> Result<(Vec<SquareMatrix>, Vec<Vector>), String> {
        const D: usize = 3;
        if partition.elements_parts().len() != self.elements.len() {
            return Err("The partition must assign every element to a subdomain.".to_string());
        }
        Ok((0..partition.number_of_parts())
            .map(|part| {
                let nodes = partition.part_nodes(part);
                let local: HashMap<usize, usize> = nodes
                    .iter()
                    .enumerate()
                    .map(|(local, &node)| (node, local))
                    .collect();
                let mut stiffness = SquareMatrix::zero(D * nodes.len());
                let mut force = Vector::zero(D * nodes.len());
                partition.part_elements(part).iter().for_each(|&element| {
                    let element = &self.elements[element];
                    let dofs: Vec<usize> = element
                        .nodes
                        .iter()
                        .flat_map(|node| {
                            let base = D * local[node];
                            (0..D).map(move |i| base + i)
                        })
                        .collect();
                    dofs.iter().enumerate().for_each(|(a, &dof_a)| {
                        force[dof_a] += element.force[a];
                        dofs.iter().enumerate().for_each(|(b, &dof_b)| {
                            stiffness[dof_a][dof_b] += element.stiffness[a][b]
                        })
                    })
                });
                (stiffness, force)
            })
            .unzip())
    }
}
