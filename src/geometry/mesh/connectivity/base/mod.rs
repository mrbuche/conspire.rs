use crate::geometry::mesh::connectivity::{Connectivity, iter::ElementIter};
use std::{fmt::Debug, num::TryFromIntError};

pub trait ConnectivityImpl {
    fn is_empty(&self) -> bool;
    fn number_of_elements(&self) -> usize;
    fn number_of_faces(&self) -> Option<usize>;
    fn number_of_faces_per_element<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>;
    fn number_of_nodes_per_element(&self) -> Option<usize>;
    fn number_of_nodes_per_face<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>;
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str;
    #[cfg(feature = "netcdf")]
    fn flat_connectivity<I>(&self) -> FlatConnectivity<I>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>;
}

#[cfg(feature = "netcdf")]
pub enum FlatConnectivity<I> {
    Primitive(Vec<I>),
    Polytopal(Vec<I>, Vec<I>),
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
    pub fn number_of_elements(&self) -> usize {
        match self {
            Connectivity::Hexahedral(c) => c.number_of_elements(),
            Connectivity::Polyhedral(c) => c.number_of_elements(),
            Connectivity::Polygonal(c) => c.number_of_elements(),
            Connectivity::Quadrilateral(c) => c.number_of_elements(),
            Connectivity::Tetrahedral(c) => c.number_of_elements(),
            Connectivity::Triangular(c) => c.number_of_elements(),
        }
    }
    pub fn number_of_faces(&self) -> Option<usize> {
        match self {
            Connectivity::Hexahedral(c) => c.number_of_faces(),
            Connectivity::Polyhedral(c) => c.number_of_faces(),
            Connectivity::Polygonal(c) => c.number_of_faces(),
            Connectivity::Quadrilateral(c) => c.number_of_faces(),
            Connectivity::Tetrahedral(c) => c.number_of_faces(),
            Connectivity::Triangular(c) => c.number_of_faces(),
        }
    }
    pub fn number_of_faces_per_element<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        match self {
            Connectivity::Hexahedral(c) => c.number_of_faces_per_element(),
            Connectivity::Polyhedral(c) => c.number_of_faces_per_element(),
            Connectivity::Polygonal(c) => c.number_of_faces_per_element(),
            Connectivity::Quadrilateral(c) => c.number_of_faces_per_element(),
            Connectivity::Tetrahedral(c) => c.number_of_faces_per_element(),
            Connectivity::Triangular(c) => c.number_of_faces_per_element(),
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
    pub fn number_of_nodes_per_face<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        match self {
            Connectivity::Hexahedral(c) => c.number_of_nodes_per_face(),
            Connectivity::Polyhedral(c) => c.number_of_nodes_per_face(),
            Connectivity::Polygonal(c) => c.number_of_nodes_per_face(),
            Connectivity::Quadrilateral(c) => c.number_of_nodes_per_face(),
            Connectivity::Tetrahedral(c) => c.number_of_nodes_per_face(),
            Connectivity::Triangular(c) => c.number_of_nodes_per_face(),
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
    pub fn flat_connectivity<I>(&self) -> FlatConnectivity<I>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        match self {
            Connectivity::Hexahedral(c) => c.flat_connectivity(),
            Connectivity::Polyhedral(c) => c.flat_connectivity(),
            Connectivity::Polygonal(c) => c.flat_connectivity(),
            Connectivity::Quadrilateral(c) => c.flat_connectivity(),
            Connectivity::Tetrahedral(c) => c.flat_connectivity(),
            Connectivity::Triangular(c) => c.flat_connectivity(),
        }
    }
}
