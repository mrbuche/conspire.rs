#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{Set, Tensor, TensorVec},
};
use std::array::from_fn;

impl<const D: usize> Mesh<D> {
    pub(crate) fn retain_elements(
        &mut self,
        mut keep: impl FnMut(usize, &[usize], &Coordinates<D>) -> bool,
    ) {
        let coordinates = self.coordinates();
        let numbers = self.blocks().map(<[usize]>::to_vec);
        let mut remap = vec![usize::MAX; coordinates.len()];
        let mut new_coordinates = Coordinates::new();
        let mut id = |node: usize| {
            if remap[node] == usize::MAX {
                remap[node] = new_coordinates.len();
                new_coordinates.push(coordinates[node].clone())
            }
            remap[node]
        };
        let mut index = 0;
        let mut blocks = Vec::with_capacity(self.number_of_element_blocks());
        for block in self.iter() {
            let kept: Vec<&[usize]> = block
                .iter()
                .filter(|element| {
                    let keeping = keep(index, element, coordinates);
                    index += 1;
                    keeping
                })
                .collect();
            blocks.push(match block {
                Connectivity::Hexahedral(_) => {
                    Connectivity::Hexahedral(primitive::<8>(&kept, &mut id).into())
                }
                Connectivity::Pyramidal(_) => {
                    Connectivity::Pyramidal(primitive::<5>(&kept, &mut id).into())
                }
                Connectivity::Quadrilateral(_) => {
                    Connectivity::Quadrilateral(primitive::<4>(&kept, &mut id).into())
                }
                Connectivity::Tetrahedral(_) => {
                    Connectivity::Tetrahedral(primitive::<4>(&kept, &mut id).into())
                }
                Connectivity::Triangular(_) => {
                    Connectivity::Triangular(primitive::<3>(&kept, &mut id).into())
                }
                Connectivity::Wedge(_) => {
                    Connectivity::Wedge(primitive::<6>(&kept, &mut id).into())
                }
                Connectivity::Polyhedral(c) => {
                    Connectivity::Polyhedral(polytopal(c.faces_nodes(), &kept, &mut id).into())
                }
                Connectivity::Polygonal(c) => {
                    Connectivity::Polygonal(polytopal(c.faces_nodes(), &kept, &mut id).into())
                }
            })
        }
        let connectivities = match numbers {
            Some(numbers) => Connectivities::from((blocks, numbers)),
            None => Connectivities::from(blocks),
        };
        *self = (connectivities, Set::from(new_coordinates)).into()
    }
}

fn primitive<const N: usize>(
    kept: &[&[usize]],
    id: &mut impl FnMut(usize) -> usize,
) -> Vec<[usize; N]> {
    kept.iter()
        .map(|element| from_fn(|i| id(element[i])))
        .collect()
}

fn polytopal(
    faces_nodes: &[Vec<usize>],
    kept: &[&[usize]],
    id: &mut impl FnMut(usize) -> usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut faces = vec![usize::MAX; faces_nodes.len()];
    let mut new_faces_nodes = Vec::new();
    let elements_faces = kept
        .iter()
        .map(|element| {
            element
                .iter()
                .map(|&face| {
                    if faces[face] == usize::MAX {
                        faces[face] = new_faces_nodes.len();
                        new_faces_nodes
                            .push(faces_nodes[face].iter().map(|&node| id(node)).collect())
                    }
                    faces[face]
                })
                .collect()
        })
        .collect();
    (elements_faces, new_faces_nodes)
}
