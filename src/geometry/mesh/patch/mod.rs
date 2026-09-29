#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    math::{Quantity, Tensor},
    units::Length,
};
use std::{
    array::from_fn,
    collections::{HashSet, VecDeque},
};

/// The elements around a seed node, and the nodes they touch.
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    pub elements: Vec<usize>,
    pub nodes: Vec<usize>,
}

impl<const D: usize> Mesh<D> {
    /// The patch around a seed node: the elements lying entirely within a
    /// radius of it whose nodes can all be reached from it along mesh edges
    /// that stay within the radius.
    ///
    /// The connectivity requirement keeps a patch from crossing a slit or thin
    /// wall that the radius alone would reach across.
    pub fn patch(&self, seed: usize, radius: Quantity<Length>) -> Patch {
        self.patches(&[seed], radius).remove(0)
    }
    /// The [patch](Mesh::patch) around each of several seed nodes.
    pub fn patches(&self, seeds: &[usize], radius: Quantity<Length>) -> Vec<Patch> {
        let radius = radius.value_as::<Length>();
        assert!(radius >= 0.0, "Patch radius must not be negative.");
        let elements: Vec<(&Connectivity, &[usize])> = self
            .iter()
            .flat_map(|block| block.iter().map(move |element| (block, element)))
            .collect();
        let nodes_elements = self.node_element_connectivity();
        let nodes_nodes = self.node_node_connectivity();
        let points: Vec<[f64; D]> = self
            .coordinates()
            .iter()
            .map(|x| from_fn(|k| x[k].value_as::<Length>()))
            .collect();
        seeds
            .iter()
            .map(|&seed| {
                assert!(
                    seed < points.len(),
                    "Patch seed must be a node of the mesh."
                );
                let inside = |node: usize| {
                    (0..D)
                        .map(|k| (points[node][k] - points[seed][k]).powi(2))
                        .sum::<f64>()
                        <= radius * radius
                };
                let mut reached = HashSet::from([seed]);
                let mut queue = VecDeque::from([seed]);
                while let Some(node) = queue.pop_front() {
                    for &next in &nodes_nodes[node] {
                        if inside(next) && reached.insert(next) {
                            queue.push_back(next);
                        }
                    }
                }
                let mut found: Vec<usize> = reached
                    .iter()
                    .flat_map(|&node| nodes_elements[node].iter().copied())
                    .collect();
                found.sort_unstable();
                found.dedup();
                found.retain(|&element| {
                    let (block, nodes) = elements[element];
                    block
                        .element_nodes(nodes)
                        .iter()
                        .all(|node| reached.contains(node))
                });
                let mut nodes: Vec<usize> = found
                    .iter()
                    .flat_map(|&element| {
                        let (block, nodes) = elements[element];
                        block.element_nodes(nodes)
                    })
                    .collect();
                nodes.sort_unstable();
                nodes.dedup();
                Patch {
                    elements: found,
                    nodes,
                }
            })
            .collect()
    }
}
