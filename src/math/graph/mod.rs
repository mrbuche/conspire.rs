#[cfg(test)]
mod test;

/// A general collection of edges, each joining some vertices.
pub trait Graph {
    /// The vertices joined by an edge.
    fn graph_vertices(&self, edge: usize) -> impl Iterator<Item = usize> + '_;
    /// The number of edges in the graph.
    fn number_of_graph_edges(&self) -> usize;
    /// The graph whose edges are this graph's vertices, each joining the edges that joined it.
    fn transpose(&self, number_of_incident_vertices: usize) -> Incidence {
        let mut transposed = vec![Vec::new(); number_of_incident_vertices];
        (0..self.number_of_graph_edges()).for_each(|edge| {
            self.graph_vertices(edge)
                .for_each(|vertex| transposed[vertex].push(edge))
        });
        Incidence::new(transposed, self.number_of_graph_edges())
    }
}

/// An owned graph, storing the vertices joined by each edge.
pub struct Incidence {
    edges: Vec<Vec<usize>>,
    number_of_vertices: usize,
}

impl Incidence {
    /// A graph with the given edges, each joining the vertices listed.
    pub fn new(edges: Vec<Vec<usize>>, number_of_vertices: usize) -> Self {
        Self {
            edges,
            number_of_vertices,
        }
    }
    /// A graph whose edges are its vertices, each joining the vertices adjacent to it.
    pub fn adjacency(edges: Vec<Vec<usize>>) -> Self {
        let number_of_vertices = edges.len();
        Self {
            edges,
            number_of_vertices,
        }
    }
    /// The total number of vertices, including those no edge joins.
    pub fn number_of_incident_vertices(&self) -> usize {
        self.number_of_vertices
    }
    /// The vertices joined by each edge.
    pub fn as_slice(&self) -> &[Vec<usize>] {
        &self.edges
    }
}

impl Graph for Incidence {
    fn graph_vertices(&self, edge: usize) -> impl Iterator<Item = usize> + '_ {
        self.edges[edge].iter().copied()
    }
    fn number_of_graph_edges(&self) -> usize {
        self.edges.len()
    }
}
