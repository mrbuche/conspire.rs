#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::Mesh,
    math::{Quantity, Tensor},
    units::Length,
};
use std::array::from_fn;

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
        let patcher = self.patcher(radius);
        let mut scratch = patcher.scratch();
        seeds
            .iter()
            .map(|&seed| patcher.patch(&mut scratch, seed))
            .collect()
    }
    pub(crate) fn patcher(&self, radius: Quantity<Length>) -> Patcher<'_, D> {
        let radius = radius.value_as::<Length>();
        assert!(radius >= 0.0, "Patch radius must not be negative.");
        let mut element_nodes = Vec::new();
        let mut pointers = vec![0];
        for block in self.iter() {
            for element in block.iter() {
                element_nodes.extend(block.element_nodes(element));
                pointers.push(element_nodes.len());
            }
        }
        Patcher {
            radius,
            points: self
                .coordinates()
                .iter()
                .map(|x| from_fn(|k| x[k].value_as()))
                .collect(),
            element_nodes,
            pointers,
            nodes_elements: self.node_element_connectivity(),
            nodes_nodes: self.node_node_connectivity(),
        }
    }
}

pub(crate) struct Patcher<'a, const D: usize> {
    radius: f64,
    points: Vec<[f64; D]>,
    element_nodes: Vec<usize>,
    pointers: Vec<usize>,
    nodes_elements: &'a [Vec<usize>],
    nodes_nodes: &'a [Vec<usize>],
}

pub(crate) struct Scratch {
    reached: Vec<usize>,
    queue: Vec<usize>,
    stamp: usize,
}

impl<const D: usize> Patcher<'_, D> {
    pub(crate) fn scratch(&self) -> Scratch {
        Scratch {
            reached: vec![0; self.points.len()],
            queue: Vec::new(),
            stamp: 0,
        }
    }
    pub(crate) fn patch(&self, scratch: &mut Scratch, seed: usize) -> Patch {
        assert!(
            seed < self.points.len(),
            "Patch seed must be a node of the mesh."
        );
        let Scratch {
            reached,
            queue,
            stamp,
        } = scratch;
        *stamp += 1;
        let stamp = *stamp;
        let nodes_of = |element: usize| {
            &self.element_nodes[self.pointers[element]..self.pointers[element + 1]]
        };
        let inside = |node: usize| {
            (0..D)
                .map(|k| (self.points[node][k] - self.points[seed][k]).powi(2))
                .sum::<f64>()
                <= self.radius * self.radius
        };
        queue.clear();
        queue.push(seed);
        reached[seed] = stamp;
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            for &next in &self.nodes_nodes[node] {
                if reached[next] != stamp && inside(next) {
                    reached[next] = stamp;
                    queue.push(next);
                }
            }
        }
        let mut found: Vec<usize> = queue
            .iter()
            .flat_map(|&node| self.nodes_elements[node].iter().copied())
            .collect();
        found.sort_unstable();
        found.dedup();
        found.retain(|&element| nodes_of(element).iter().all(|&node| reached[node] == stamp));
        let mut nodes: Vec<usize> = found
            .iter()
            .flat_map(|&element| nodes_of(element).iter().copied())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        Patch {
            elements: found,
            nodes,
        }
    }
}
