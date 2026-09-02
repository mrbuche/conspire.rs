use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Orthotree,
            node::{Node, cell::Cell, slot::Slot},
            subdivide::insert_bit,
        },
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{array::from_fn, collections::HashMap};

#[cfg(test)]
mod test;

type Lines = Vec<HashMap<[usize; 2], Vec<(usize, usize)>>>;

/// How a leaf's facet meets the rest of the tree.
pub(crate) enum Facet {
    /// No neighbor link recorded.
    Absent,
    /// On the root boundary.
    Boundary,
    /// A same-level or coarser leaf.
    Neighbor(usize),
    /// The finer leaves covering it.
    Refined(Vec<usize>),
}

/// The nodes of a tree's leaf corners, and the facet polygons they form.
pub(crate) struct Facets<const D: usize> {
    coordinates: Coordinates<D>,
    lines: Lines,
    nodes: HashMap<[usize; D], usize>,
}

pub(crate) fn corner_length<const D: usize, const M: usize, const N: usize, T, U, V>(
    node: &Node<D, M, N, T, U, V>,
) -> ([usize; D], usize)
where
    T: Cell,
{
    (
        from_fn(|axis| node.corner[axis].cells()),
        node.length.cells(),
    )
}

pub(crate) fn leaves<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
) -> (Vec<usize>, Vec<usize>)
where
    T: Cell,
{
    let mut element_of = vec![usize::MAX; tree.nodes.len()];
    let leaves: Vec<usize> = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.is_leaf())
        .map(|(index, _)| index)
        .collect();
    leaves
        .iter()
        .enumerate()
        .for_each(|(element, &index)| element_of[index] = element);
    (leaves, element_of)
}

fn gather<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
    index: usize,
    facet: usize,
    leaves: &mut Vec<usize>,
) where
    T: Cell,
    U: Slot,
{
    match tree.nodes[index].orthants() {
        None => leaves.push(index),
        Some(orthants) => (0..L).for_each(|k| {
            gather(
                tree,
                orthants[insert_bit(k, facet >> 1, facet & 1)].slot(),
                facet,
                leaves,
            )
        }),
    }
}

pub(crate) fn facet<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>(
    tree: &Orthotree<D, L, M, N, T, U, V>,
    index: usize,
    facet: usize,
) -> Facet
where
    T: Cell,
    U: Slot,
{
    let (axis, side) = (facet >> 1, facet & 1);
    let (corner, length) = corner_length(&tree.nodes[index]);
    match tree.nodes[index].facets[facet] {
        Some(neighbor) => {
            let neighbor = neighbor.slot();
            if tree.nodes[neighbor].is_tree() {
                let mut fine = Vec::new();
                gather(tree, neighbor, facet ^ 1, &mut fine);
                Facet::Refined(fine)
            } else {
                Facet::Neighbor(neighbor)
            }
        }
        None => {
            let root = &tree.nodes[0];
            let lo = root.corner[axis].cells();
            let bound = if side == 0 {
                lo
            } else {
                lo + root.length.cells()
            };
            if corner[axis] + side * length == bound {
                Facet::Boundary
            } else {
                Facet::Absent
            }
        }
    }
}

impl<const D: usize> Facets<D> {
    pub(crate) fn new<const L: usize, const M: usize, const N: usize, T, U, V>(
        tree: &Orthotree<D, L, M, N, T, U, V>,
        leaves: &[usize],
    ) -> Self
    where
        T: Cell,
    {
        let mut nodes = HashMap::<[usize; D], usize>::new();
        let mut coordinates = Coordinates::<D>::new();
        leaves.iter().for_each(|&index| {
            let (corner, length) = corner_length(&tree.nodes[index]);
            (0..N).for_each(|k| {
                let key: [usize; D] = from_fn(|axis| corner[axis] + ((k >> axis) & 1) * length);
                nodes.entry(key).or_insert_with(|| {
                    coordinates.push(from_fn(|axis| key[axis] as Scalar).into());
                    coordinates.len() - 1
                });
            })
        });
        let mut lines: Lines = vec![HashMap::new(); D];
        if D == 3 {
            nodes.iter().for_each(|(key, &node)| {
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
        Self {
            coordinates,
            lines,
            nodes,
        }
    }
    #[cfg(test)]
    pub(crate) fn coordinates(&self) -> &Coordinates<D> {
        &self.coordinates
    }
    pub(crate) fn into_coordinates(self) -> Coordinates<D> {
        self.coordinates
    }
    #[allow(dead_code)]
    pub(crate) fn corners<const N: usize>(&self, corner: [usize; D], length: usize) -> [usize; N] {
        from_fn(|k| {
            self.nodes[&from_fn::<_, D, _>(|axis| corner[axis] + ((k >> axis) & 1) * length)]
        })
    }
    /// The polygon of the facet of a cell of the given `corner` and `size`,
    /// lying in `plane` normal to `axis`, with the hanging nodes of any finer
    /// neighbors inserted into its node loop.
    pub(crate) fn polygon(
        &self,
        corner: [usize; D],
        size: usize,
        axis: usize,
        plane: usize,
        flip: bool,
    ) -> Vec<usize> {
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
            face.push(self.nodes[&key]);
            key[tangent] = finish;
            face.push(self.nodes[&key]);
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
                face.push(self.nodes[&key]);
                let (tangent, from, unto) = if ua != ub {
                    (tangents[0], span[0] + ua * size, span[0] + ub * size)
                } else {
                    (tangents[1], span[1] + va * size, span[1] + vb * size)
                };
                if let Some(line) =
                    self.lines[tangent].get(&from_fn::<_, 2, _>(|j| key[(tangent + 1 + j) % D]))
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
        face
    }
    /// The outward polygons covering each facet of a leaf: one per facet, or
    /// the finer neighbors' when that facet is refined.
    #[allow(dead_code)]
    pub(crate) fn leaf_polygons<const L: usize, const M: usize, const N: usize, T, U, V>(
        &self,
        tree: &Orthotree<D, L, M, N, T, U, V>,
        index: usize,
    ) -> [Vec<Vec<usize>>; M]
    where
        T: Cell,
        U: Slot,
    {
        let (corner, length) = corner_length(&tree.nodes[index]);
        from_fn(|f| {
            let (axis, side) = (f >> 1, f & 1);
            let plane = corner[axis] + side * length;
            let flip = side == 0;
            match facet(tree, index, f) {
                Facet::Refined(fine) => fine
                    .into_iter()
                    .map(|leaf| {
                        let (fine_corner, fine_length) = corner_length(&tree.nodes[leaf]);
                        self.polygon(fine_corner, fine_length, axis, plane, flip)
                    })
                    .collect(),
                _ => vec![self.polygon(corner, length, axis, plane, flip)],
            }
        })
    }
}
