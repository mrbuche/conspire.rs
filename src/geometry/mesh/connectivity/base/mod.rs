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
    fn as_impl(&self) -> &dyn ConnectivityImpl {
        match self {
            Connectivity::Hexahedral(c) => c,
            Connectivity::Polyhedral(c) => c,
            Connectivity::Polygonal(c) => c,
            Connectivity::Quadrilateral(c) => c,
            Connectivity::Tetrahedral(c) => c,
            Connectivity::Triangular(c) => c,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.as_impl().is_empty()
    }
    pub fn iter(&self) -> ElementIter<'_> {
        self.into_iter()
    }
    pub fn len(&self) -> usize {
        self.as_impl().len()
    }
    pub fn number_of_nodes_per_element(&self) -> Option<usize> {
        self.as_impl().number_of_nodes_per_element()
    }
    #[cfg(feature = "netcdf")]
    pub fn exodus_element_type(&self) -> &str {
        self.as_impl().exodus_element_type()
    }
    #[cfg(feature = "netcdf")]
    pub fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        self.as_impl().primitive_connectivity_flattened()
    }
}
