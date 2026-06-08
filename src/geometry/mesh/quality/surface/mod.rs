#[cfg(test)]
mod test;

use crate::geometry::mesh::Mesh;
use std::collections::HashMap;

impl<const D: usize> Mesh<D> {
    pub fn boundary_edges(&self) -> Vec<[usize; 2]> {
        self.edge_incidence()
            .into_values()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect()
    }
    fn edge_incidence(&self) -> HashMap<[usize; 2], ([usize; 2], usize)> {
        let mut edges = HashMap::new();
        self.iter().for_each(|block| {
            let local_edges = block.local_faces();
            block.iter().for_each(|element| {
                local_edges.iter().for_each(|edge| {
                    let oriented = [element[edge[0]], element[edge[1]]];
                    let key = if oriented[0] < oriented[1] {
                        oriented
                    } else {
                        [oriented[1], oriented[0]]
                    };
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
