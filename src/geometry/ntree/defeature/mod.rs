#[cfg(test)]
mod test;

use crate::geometry::ntree::{Orthotree, node::Kind, subdivide::insert_bit};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    hash::Hash,
};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Copy + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy + Eq + Hash,
{
    pub fn defeature(&mut self, minimum: usize) {
        let count = self.len();
        let mut parent: Vec<usize> = (0..count).collect();
        let leaves: Vec<usize> = (0..count)
            .filter(|&i| self.nodes[i].is_leaf() && self.nodes[i].value.is_some())
            .collect();
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
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
                                edges.push((leaf, other, length.min(span).pow((D - 1) as u32)));
                            }
                        }
                    }
                }
            }
        }
        let mut volume: HashMap<usize, usize> = HashMap::new();
        let mut value: HashMap<usize, V> = HashMap::new();
        for &leaf in &leaves {
            let root = find(&mut parent, leaf);
            let length: usize = self.nodes[leaf].length.into();
            *volume.entry(root).or_default() += length.pow(D as u32);
            value
                .entry(root)
                .or_insert_with(|| self.nodes[leaf].value.unwrap());
        }
        let mut adjacency: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        for (leaf, other, span) in edges {
            let (a, b) = (find(&mut parent, leaf), find(&mut parent, other));
            if a != b {
                *adjacency.entry(a).or_default().entry(b).or_default() += span;
                *adjacency.entry(b).or_default().entry(a).or_default() += span;
            }
        }
        let mut queue: BinaryHeap<Reverse<(usize, usize)>> = volume
            .iter()
            .filter(|&(_, &size)| size < minimum)
            .map(|(&root, &size)| Reverse((size, root)))
            .collect();
        while let Some(Reverse((size, root))) = queue.pop() {
            if value.get(&root).is_none_or(|_| volume[&root] != size) || size >= minimum {
                continue;
            }
            let neighbors = match adjacency.get(&root) {
                Some(map) if !map.is_empty() => map,
                _ => continue,
            };
            let mut by_value: HashMap<V, usize> = HashMap::new();
            for (other, &span) in neighbors {
                *by_value.entry(value[other]).or_default() += span;
            }
            let into = *by_value.iter().max_by_key(|&(_, &span)| span).unwrap().0;
            let group: Vec<usize> = neighbors
                .keys()
                .copied()
                .filter(|other| value[other] == into)
                .chain([root])
                .collect();
            group[..group.len() - 1]
                .iter()
                .for_each(|&other| union(&mut parent, root, other));
            let root = find(&mut parent, root);
            let merged_volume = group.iter().map(|node| volume[node]).sum();
            let mut merged: HashMap<usize, usize> = HashMap::new();
            for node in &group {
                if let Some(map) = adjacency.remove(node) {
                    for (other, span) in map {
                        let other = find(&mut parent, other);
                        if other != root {
                            *merged.entry(other).or_default() += span;
                        }
                    }
                }
            }
            group.iter().filter(|&&node| node != root).for_each(|node| {
                volume.remove(node);
                value.remove(node);
            });
            for other in merged.keys() {
                let map = adjacency.get_mut(other).unwrap();
                let span: usize = group.iter().filter_map(|node| map.remove(node)).sum();
                map.insert(root, span);
            }
            volume.insert(root, merged_volume);
            value.insert(root, into);
            adjacency.insert(root, merged);
            if merged_volume < minimum {
                queue.push(Reverse((merged_volume, root)));
            }
        }
        leaves.iter().for_each(|&leaf| {
            self.nodes[leaf].value = Some(value[&find(&mut parent, leaf)]);
        });
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
