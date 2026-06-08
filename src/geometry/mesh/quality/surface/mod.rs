#[cfg(test)]
mod test;

use crate::geometry::mesh::Mesh;
use std::collections::HashMap;

impl<const D: usize> Mesh<D> {
    pub fn boundary_edges(&self) -> Vec<Vec<usize>> {
        self.edge_incidence()
            .into_values()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect()
    }
    fn edge_incidence(&self) -> HashMap<Vec<usize>, (Vec<usize>, usize)> {
        let mut edges = HashMap::new();
        self.iter().for_each(|block| {
            let local_edges = block.local_faces();
            block.iter().for_each(|element| {
                local_edges.iter().for_each(|edge| {
                    let oriented: Vec<usize> = edge.iter().map(|&local| element[local]).collect();
                    let mut key = oriented.clone();
                    key.sort_unstable();
                    edges
                        .entry(key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((oriented, 1));
                })
            })
        });
        edges
    }
}
