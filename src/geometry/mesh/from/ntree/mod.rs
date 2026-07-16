#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
        ntree::{Octree, Orthotree, Quadtree, node::Node, subdivide::insert_bit},
    },
    math::Scalar,
};
use std::{array::from_fn, collections::HashMap};

impl<T, U, V> From<Octree<T, U, V>> for Mesh<3>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
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
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
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

fn corner_length<const D: usize, const M: usize, const N: usize, T, U, V>(
    node: &Node<D, M, N, T, U, V>,
) -> ([usize; D], usize)
where
    T: Copy + Into<usize>,
{
    (from_fn(|axis| node.corner[axis].into()), node.length.into())
}

fn gather<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
    index: usize,
    facet: usize,
    leaves: &mut Vec<usize>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    match tree.nodes[index].orthants() {
        None => leaves.push(index),
        Some(orthants) => (0..L).for_each(|k| {
            gather(
                tree,
                orthants[insert_bit(k, facet >> 1, facet & 1)].into(),
                facet,
                leaves,
            )
        }),
    }
}

fn polytopes<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Coordinates<D>)
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut element_of = vec![usize::MAX; tree.nodes.len()];
    let leaves = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.is_leaf())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    leaves
        .iter()
        .enumerate()
        .for_each(|(element, &index)| element_of[index] = element);
    let mut nodes_map = HashMap::<[usize; D], usize>::new();
    let mut coordinates = Vec::<Coordinate<D>>::new();
    leaves.iter().for_each(|&index| {
        let (corner, length) = corner_length(&tree.nodes[index]);
        (0..N).for_each(|k| {
            let key: [usize; D] = from_fn(|axis| corner[axis] + ((k >> axis) & 1) * length);
            nodes_map.entry(key).or_insert_with(|| {
                coordinates.push(from_fn(|axis| key[axis] as Scalar).into());
                coordinates.len() - 1
            });
        })
    });
    let mut lines = vec![HashMap::<[usize; 2], Vec<(usize, usize)>>::new(); D];
    if D == 3 {
        nodes_map.iter().for_each(|(key, &node)| {
            (0..D).for_each(|axis| {
                lines[axis]
                    .entry(from_fn(|j| key[(axis + 1 + j) % D]))
                    .or_default()
                    .push((key[axis], node))
            })
        });
        lines
            .iter_mut()
            .for_each(|map| map.values_mut().for_each(|line| line.sort_unstable()));
    }
    let mut elements_faces = vec![Vec::new(); leaves.len()];
    let mut faces_nodes = Vec::<Vec<usize>>::new();
    let mut emit = |corner: [usize; D],
                    size: usize,
                    axis: usize,
                    plane: usize,
                    elements: &[usize],
                    flip: bool| {
        let mut face = Vec::new();
        if D == 2 {
            let tangent = 1 - axis;
            let (start, finish) = if axis == 0 {
                (corner[tangent], corner[tangent] + size)
            } else {
                (corner[tangent] + size, corner[tangent])
            };
            let mut key = corner;
            key[axis] = plane;
            key[tangent] = start;
            face.push(nodes_map[&key]);
            key[tangent] = finish;
            face.push(nodes_map[&key]);
        } else if D == 3 {
            let tangents = [(axis + 1) % D, (axis + 2) % D];
            let span = [corner[tangents[0]], corner[tangents[1]]];
            let quad = [(0, 0), (1, 0), (1, 1), (0, 1)];
            (0..4).for_each(|k| {
                let (ua, va) = quad[k];
                let (ub, vb) = quad[(k + 1) % 4];
                let mut key = [0; D];
                key[axis] = plane;
                key[tangents[0]] = span[0] + ua * size;
                key[tangents[1]] = span[1] + va * size;
                face.push(nodes_map[&key]);
                let (tangent, from, unto) = if ua != ub {
                    (tangents[0], span[0] + ua * size, span[0] + ub * size)
                } else {
                    (tangents[1], span[1] + va * size, span[1] + vb * size)
                };
                if let Some(line) =
                    lines[tangent].get(&from_fn::<_, 2, _>(|j| key[(tangent + 1 + j) % D]))
                {
                    let (min, max) = (from.min(unto), from.max(unto));
                    let interior = line
                        .iter()
                        .filter(|(position, _)| *position > min && *position < max)
                        .map(|&(_, node)| node);
                    if unto > from {
                        face.extend(interior)
                    } else {
                        face.extend(interior.rev())
                    }
                }
            });
        } else {
            unimplemented!()
        }
        if flip {
            face.reverse();
        }
        let index = faces_nodes.len();
        faces_nodes.push(face);
        elements
            .iter()
            .for_each(|&element| elements_faces[element].push(index));
    };
    let root = &tree.nodes[0];
    let lo: [usize; D] = from_fn(|axis| root.corner[axis].into());
    let hi: [usize; D] = from_fn(|axis| lo[axis] + root.length.into());
    for &index in &leaves {
        let element = element_of[index];
        let (corner, length) = corner_length(&tree.nodes[index]);
        for facet in 0..M {
            let (axis, side) = (facet >> 1, facet & 1);
            let plane = corner[axis] + side * length;
            match tree.nodes[index].facets[facet] {
                Some(neighbor) => {
                    let neighbor = neighbor.into();
                    if tree.nodes[neighbor].is_tree() {
                        let mut fine = Vec::new();
                        gather(tree, neighbor, facet ^ 1, &mut fine);
                        fine.into_iter().for_each(|leaf| {
                            let (fine_corner, fine_length) = corner_length(&tree.nodes[leaf]);
                            emit(
                                fine_corner,
                                fine_length,
                                axis,
                                plane,
                                &[element_of[leaf], element],
                                false,
                            )
                        })
                    } else if side == 0 {
                        emit(
                            corner,
                            length,
                            axis,
                            plane,
                            &[element_of[neighbor], element],
                            false,
                        )
                    }
                }
                None => {
                    if plane == if side == 0 { lo[axis] } else { hi[axis] } {
                        emit(corner, length, axis, plane, &[element], side == 0)
                    }
                }
            }
        }
    }
    (elements_faces, faces_nodes, coordinates.into())
}
