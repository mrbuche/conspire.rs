use crate::geometry::mesh::connectivity::{Connectivity, iter::ElementIter};

pub trait ConnectivityImpl {
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn number_of_nodes_per_element(&self) -> Option<usize>;
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str;
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>>;
}

impl Connectivity {
    pub fn is_empty(&self) -> bool {
        match self {
            Connectivity::Hexahedral(c) => c.is_empty(),
            Connectivity::Polyhedral(c) => c.is_empty(),
            Connectivity::Polygonal(c) => c.is_empty(),
            Connectivity::Quadrilateral(c) => c.is_empty(),
            Connectivity::Tetrahedral(c) => c.is_empty(),
            Connectivity::Triangular(c) => c.is_empty(),
        }
    }
    pub fn iter(&self) -> ElementIter<'_> {
        self.into_iter()
    }
    pub fn len(&self) -> usize {
        match self {
            Connectivity::Hexahedral(c) => c.len(),
            Connectivity::Polyhedral(c) => c.len(),
            Connectivity::Polygonal(c) => c.len(),
            Connectivity::Quadrilateral(c) => c.len(),
            Connectivity::Tetrahedral(c) => c.len(),
            Connectivity::Triangular(c) => c.len(),
        }
    }
    pub fn number_of_nodes_per_element(&self) -> Option<usize> {
        match self {
            Connectivity::Hexahedral(c) => c.number_of_nodes_per_element(),
            Connectivity::Polyhedral(c) => c.number_of_nodes_per_element(),
            Connectivity::Polygonal(c) => c.number_of_nodes_per_element(),
            Connectivity::Quadrilateral(c) => c.number_of_nodes_per_element(),
            Connectivity::Tetrahedral(c) => c.number_of_nodes_per_element(),
            Connectivity::Triangular(c) => c.number_of_nodes_per_element(),
        }
    }
    #[cfg(feature = "netcdf")]
    pub fn exodus_element_type(&self) -> &str {
        match self {
            Connectivity::Hexahedral(c) => c.exodus_element_type(),
            Connectivity::Polyhedral(c) => c.exodus_element_type(),
            Connectivity::Polygonal(c) => c.exodus_element_type(),
            Connectivity::Quadrilateral(c) => c.exodus_element_type(),
            Connectivity::Tetrahedral(c) => c.exodus_element_type(),
            Connectivity::Triangular(c) => c.exodus_element_type(),
        }
    }
    #[cfg(feature = "netcdf")]
    pub fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        match self {
            Connectivity::Hexahedral(c) => c.primitive_connectivity_flattened(),
            Connectivity::Polyhedral(c) => c.primitive_connectivity_flattened(),
            Connectivity::Polygonal(c) => c.primitive_connectivity_flattened(),
            Connectivity::Quadrilateral(c) => c.primitive_connectivity_flattened(),
            Connectivity::Tetrahedral(c) => c.primitive_connectivity_flattened(),
            Connectivity::Triangular(c) => c.primitive_connectivity_flattened(),
        }
    }
}
