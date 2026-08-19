#[cfg(test)]
mod test;

use crate::geometry::ntree::{Orthotree, node::Kind, subdivide::insert_bit};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
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
        loop {
            let protruded = self.reduce_protrusions();
            let clustered = self.reduce_clusters(minimum);
            if !protruded && !clustered {
                break;
            }
        }
    }
    /// Reassigns leaves whose value differs from neighboring leaves covering
    /// at least `M - 1` of their `M` facets' worth of area, absorbing
    /// single-cell protrusions into the majority-neighboring value.
    fn reduce_protrusions(&mut self) -> bool {
        let (leaves, pairs) = self.leaf_pairs();
        let mut differing: HashMap<usize, HashMap<V, usize>> = HashMap::new();
        for (a, b, weight) in pairs {
            let (value_a, value_b) = (self.nodes[a].value.unwrap(), self.nodes[b].value.unwrap());
            if value_a != value_b {
                *differing.entry(a).or_default().entry(value_b).or_default() += weight;
                *differing.entry(b).or_default().entry(value_a).or_default() += weight;
            }
        }
        let mut reassignments: Vec<(usize, V)> = Vec::new();
        for &leaf in &leaves {
            let Some(neighbors) = differing.get(&leaf) else {
                continue;
            };
            let length: usize = self.nodes[leaf].length.into();
            let facet_area = length.pow((D - 1) as u32);
            let differing_area: usize = neighbors.values().sum();
            if differing_area >= (M - 1) * facet_area {
                let into = *neighbors.iter().max_by_key(|&(_, &area)| area).unwrap().0;
                reassignments.push((leaf, into));
            }
        }
        let changed = !reassignments.is_empty();
        reassignments.into_iter().for_each(|(leaf, into)| {
            self.nodes[leaf].value = Some(into);
        });
        changed
    }
    /// Merges connected clusters of leaves with total volume below `minimum`
    /// into the neighboring cluster with which they share the most facet
    /// area.
    fn reduce_clusters(&mut self, minimum: usize) -> bool {
        let count = self.len();
        let mut parent: Vec<usize> = (0..count).collect();
        let (leaves, pairs) = self.leaf_pairs();
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for (a, b, weight) in pairs {
            if self.nodes[a].value == self.nodes[b].value {
                union(&mut parent, a, b);
            } else {
                edges.push((a, b, weight));
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
        let mut changed = false;
        while let Some(Reverse((size, root))) = queue.pop() {
            if value.get(&root).is_none_or(|_| volume[&root] != size) || size >= minimum {
                continue;
            }
            let neighbors = match adjacency.get(&root) {
                Some(map) if !map.is_empty() => map,
                _ => continue,
            };
            changed = true;
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
        changed
    }
    /// Returns every valued leaf and every (deduplicated) pair of
    /// facet-adjacent valued leaves, weighted by shared facet area.
    fn leaf_pairs(&self) -> (Vec<usize>, Vec<(usize, usize, usize)>) {
        let leaves: Vec<usize> = (0..self.len())
            .filter(|&i| self.nodes[i].is_leaf() && self.nodes[i].value.is_some())
            .collect();
        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
        for &leaf in &leaves {
            let length: usize = self.nodes[leaf].length.into();
            for face in 0..M {
                if let Some(neighbor) = self.nodes[leaf].facets[face] {
                    let mut others = Vec::new();
                    self.face_leaves(neighbor.into(), face ^ 1, &mut others);
                    for other in others {
                        if self.nodes[other].value.is_some() {
                            let key = if leaf < other {
                                (leaf, other)
                            } else {
                                (other, leaf)
                            };
                            if visited.insert(key) {
                                let span: usize = self.nodes[other].length.into();
                                pairs.push((leaf, other, length.min(span).pow((D - 1) as u32)));
                            }
                        }
                    }
                }
            }
        }
        (leaves, pairs)
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
