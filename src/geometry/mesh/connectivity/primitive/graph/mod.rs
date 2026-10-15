use crate::{geometry::mesh::connectivity::primitive::PrimitiveConnectivity, math::Graph};

impl<const M: usize, const N: usize> Graph for PrimitiveConnectivity<M, N> {
    fn number_of_graph_edges(&self) -> usize {
        self.0.members().len()
    }
    fn graph_vertices(&self, edge: usize) -> impl Iterator<Item = usize> + '_ {
        self.0.members()[edge].iter().copied()
    }
}
