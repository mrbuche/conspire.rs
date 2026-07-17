use crate::math::Set;

/// A connected set of members or nodes.
pub struct Graph {
    adjacency: Set<Vec<Vec<usize>>>,
}

impl Graph {
    pub fn adjacency(&self) -> &[Vec<usize>] {
        self.adjacency.members()
    }
}

impl From<Vec<Vec<usize>>> for Graph {
    fn from(data: Vec<Vec<usize>>) -> Self {
        let adjacency = data.into();
        Self { adjacency }
    }
}
