#[cfg(test)]
mod test;

pub(crate) mod facets;

use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivity, Mesh,
        from::ntree::facets::{Facet, Facets, corner_length, facet, leaves},
    },
    ntree::{
        Octree, Orthotree, Quadtree,
        node::{cell::Cell, slot::Slot},
    },
};

impl<T, U, V> From<Octree<T, U, V>> for Mesh<3>
where
    T: Cell,
    U: Slot,
{
    fn from(octree: Octree<T, U, V>) -> Self {
        let (elements_faces, faces_nodes, mut coordinates) = polytopes(&octree);
        octree.rescale_coordinates(&mut coordinates);
        (
            vec![Connectivity::Polyhedral(
                (elements_faces, faces_nodes).into(),
            )],
            coordinates,
        )
            .into()
    }
}

impl<T, U, V> From<Quadtree<T, U, V>> for Mesh<2>
where
    T: Cell,
    U: Slot,
{
    fn from(quadtree: Quadtree<T, U, V>) -> Self {
        let (elements_faces, faces_nodes, mut coordinates) = polytopes(&quadtree);
        quadtree.rescale_coordinates(&mut coordinates);
        (
            vec![Connectivity::Polygonal(
                (elements_faces, faces_nodes).into(),
            )],
            coordinates,
        )
            .into()
    }
}

fn polytopes<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Coordinates<D>)
where
    T: Cell,
    U: Slot,
{
    let (leaves, element_of) = leaves(tree);
    let facets = Facets::<D>::new::<L, M, N, T, U, V>(tree, &leaves);
    let mut elements_faces = vec![Vec::new(); leaves.len()];
    let mut faces_nodes = Vec::<Vec<usize>>::new();
    let mut emit = |corner: [usize; D],
                    size: usize,
                    axis: usize,
                    plane: usize,
                    elements: &[usize],
                    flip: bool| {
        let index = faces_nodes.len();
        faces_nodes.push(facets.polygon(corner, size, axis, plane, flip));
        elements
            .iter()
            .for_each(|&element| elements_faces[element].push(index));
    };
    for &index in &leaves {
        let element = element_of[index];
        let (corner, length) = corner_length(&tree.nodes[index]);
        for f in 0..M {
            let (axis, side) = (f >> 1, f & 1);
            let plane = corner[axis] + side * length;
            match facet(tree, index, f) {
                Facet::Refined(fine) => fine.into_iter().for_each(|leaf| {
                    let (fine_corner, fine_length) = corner_length(&tree.nodes[leaf]);
                    let fine_element = element_of[leaf];
                    let flip = if fine_element < element {
                        side == 1
                    } else {
                        side == 0
                    };
                    emit(
                        fine_corner,
                        fine_length,
                        axis,
                        plane,
                        &[fine_element, element],
                        flip,
                    )
                }),
                Facet::Neighbor(neighbor) => {
                    if side == 0 {
                        let neighbor_element = element_of[neighbor];
                        emit(
                            corner,
                            length,
                            axis,
                            plane,
                            &[neighbor_element, element],
                            element < neighbor_element,
                        )
                    }
                }
                Facet::Boundary => emit(corner, length, axis, plane, &[element], side == 0),
                Facet::Absent => {}
            }
        }
    }
    (elements_faces, faces_nodes, facets.into_coordinates())
}
