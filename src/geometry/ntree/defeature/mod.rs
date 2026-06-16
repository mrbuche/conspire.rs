#[cfg(test)]
mod test;

use crate::geometry::{
    grid::{Pixels, Voxels},
    ntree::{Octree, Orthotree, Quadtree, node::Kind, subdivide::insert_bit},
};
use std::{collections::HashMap, hash::Hash};

impl<V: Copy + Eq + Hash> Pixels<V> {
    pub fn defeature(self, minimum: usize) -> Self {
        let mut quadtree = Quadtree::<u16, usize, V>::from(self);
        quadtree.defeature(minimum);
        (&quadtree).into()
    }
}

impl<V: Copy + Eq + Hash> Voxels<V> {
    pub fn defeature(self, minimum: usize) -> Self {
        let mut octree = Octree::<u16, usize, V>::from(self);
        octree.defeature(minimum);
        (&octree).into()
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Copy + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy + Eq + Hash,
{
    pub fn defeature(&mut self, minimum: usize) {
        let count = self.len();
        loop {
            let mut parent: Vec<usize> = (0..count).collect();
            let leaves: Vec<usize> = (0..count)
                .filter(|&i| self.nodes[i].is_leaf() && self.nodes[i].value.is_some())
                .collect();
            let mut cross: Vec<(usize, V, usize)> = Vec::new();
            for &leaf in &leaves {
                let value = self.nodes[leaf].value.unwrap();
                let length: usize = self.nodes[leaf].length.into();
                for face in 0..M {
                    if let Some(neighbor) = self.nodes[leaf].facets[face] {
                        let mut others = Vec::new();
                        self.face_leaves(neighbor.into(), face ^ 1, &mut others);
                        for other in others {
                            if let Some(adjacent) = self.nodes[other].value {
                                if adjacent == value {
                                    union(&mut parent, leaf, other);
                                } else {
                                    let span: usize = self.nodes[other].length.into();
                                    let area = length.min(span).pow((D - 1) as u32);
                                    cross.push((leaf, adjacent, area));
                                    cross.push((other, value, area));
                                }
                            }
                        }
                    }
                }
            }
            let mut volume: HashMap<usize, usize> = HashMap::new();
            for &leaf in &leaves {
                let length: usize = self.nodes[leaf].length.into();
                *volume.entry(find(&mut parent, leaf)).or_default() += length.pow(D as u32);
            }
            let mut area: HashMap<(usize, V), usize> = HashMap::new();
            for (leaf, adjacent, span) in cross {
                *area.entry((find(&mut parent, leaf), adjacent)).or_default() += span;
            }
            let target = volume
                .iter()
                .filter(|&(_, &size)| size < minimum)
                .filter_map(|(&root, &size)| {
                    area.iter()
                        .filter(|&(&(other, _), _)| other == root)
                        .max_by_key(|&(_, &span)| span)
                        .map(|(&(_, value), _)| (size, root, value))
                })
                .min_by_key(|&(size, ..)| size);
            match target {
                Some((_, root, value)) => leaves
                    .iter()
                    .filter(|&&leaf| find(&mut parent, leaf) == root)
                    .for_each(|&leaf| self.nodes[leaf].value = Some(value)),
                None => break,
            }
        }
    }
    fn face_leaves(&self, index: usize, face: usize, out: &mut Vec<usize>) {
        match &self.nodes[index].kind {
            Kind::Leaf => out.push(index),
            Kind::Tree(orthants) => {
                let (axis, side) = (face >> 1, face & 1);
                for i in 0..L {
                    let child = orthants[insert_bit(i, axis, side)].into();
                    self.face_leaves(child, face, out);
                }
            }
        }
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (find(parent, a), find(parent, b));
    if a != b {
        parent[a] = b;
    }
}
