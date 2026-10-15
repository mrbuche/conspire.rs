use crate::{geometry::mesh::connectivity::polytopal::PolytopalConnectivity, math::Graph};

impl<const M: usize> Graph for PolytopalConnectivity<M> {
    fn number_of_graph_edges(&self) -> usize {
        self.0.len()
    }
    fn graph_vertices(&self, edge: usize) -> impl Iterator<Item = usize> + '_ {
        let mut nodes: Vec<usize> = self.0[edge]
            .iter()
            .flat_map(|&face| self.1[face].iter().copied())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.into_iter()
    }
}
