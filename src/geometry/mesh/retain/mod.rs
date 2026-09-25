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
        keep: impl FnMut(usize, &[usize], &Coordinates<D>) -> bool,
    ) {
        *self = self.retained_elements(keep).0
    }
    pub(crate) fn retained_elements(
        &self,
        mut keep: impl FnMut(usize, &[usize], &Coordinates<D>) -> bool,
    ) -> (Self, Vec<usize>) {
        let coordinates = self.coordinates();
        let numbers = self.blocks().map(<[usize]>::to_vec);
        let mut remap = vec![usize::MAX; coordinates.len()];
        let mut old_nodes = Vec::new();
        let mut new_coordinates = Coordinates::new();
        let mut id = |node: usize| {
            if remap[node] == usize::MAX {
                remap[node] = new_coordinates.len();
                old_nodes.push(node);
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
            blocks.push(subset(block, &kept, &mut id))
        }
        let connectivities = match numbers {
            Some(numbers) => Connectivities::from((blocks, numbers)),
            None => Connectivities::from(blocks),
        };
        (
            (connectivities, Set::from(new_coordinates)).into(),
            old_nodes,
        )
    }
}

pub(super) fn subset(
    block: &Connectivity,
    kept: &[&[usize]],
    id: &mut impl FnMut(usize) -> usize,
) -> Connectivity {
    match block {
        Connectivity::Hexahedral(_) => Connectivity::Hexahedral(primitive::<8>(kept, id).into()),
        Connectivity::Pyramidal(_) => Connectivity::Pyramidal(primitive::<5>(kept, id).into()),
        Connectivity::Quadrilateral(_) => {
            Connectivity::Quadrilateral(primitive::<4>(kept, id).into())
        }
        Connectivity::Tetrahedral(_) => Connectivity::Tetrahedral(primitive::<4>(kept, id).into()),
        Connectivity::Triangular(_) => Connectivity::Triangular(primitive::<3>(kept, id).into()),
        Connectivity::Wedge(_) => Connectivity::Wedge(primitive::<6>(kept, id).into()),
        Connectivity::Polyhedral(c) => {
            Connectivity::Polyhedral(polytopal(c.faces_nodes(), kept, id).into())
        }
        Connectivity::Polygonal(c) => {
            Connectivity::Polygonal(polytopal(c.faces_nodes(), kept, id).into())
        }
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
