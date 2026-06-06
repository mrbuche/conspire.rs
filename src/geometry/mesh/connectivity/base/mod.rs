use crate::geometry::mesh::connectivity::{Connectivities, Connectivity, iter::ElementIter};
use std::{fmt::Debug, num::TryFromIntError};

pub trait ConnectivityImpl {
    fn is_empty(&self) -> bool;
    fn element_numbers(&self) -> Option<&[usize]>;
    fn node_element_connectivity(&self) -> &[Vec<usize>];
    fn number_elements(&mut self, numbers: Vec<usize>);
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
    pub fn local_faces(&self) -> &'static [&'static [usize]] {
        match self {
            Connectivity::Hexahedral(_) => &[
                &[0, 3, 2, 1],
                &[4, 5, 6, 7],
                &[0, 1, 5, 4],
                &[1, 2, 6, 5],
                &[2, 3, 7, 6],
                &[3, 0, 4, 7],
            ],
            Connectivity::Tetrahedral(_) => &[&[0, 2, 1], &[0, 1, 3], &[1, 2, 3], &[2, 0, 3]],
            Connectivity::Quadrilateral(_) => &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
            Connectivity::Triangular(_) => &[&[0, 1], &[1, 2], &[2, 0]],
            Connectivity::Polygonal(_) | Connectivity::Polyhedral(_) => todo!(),
        }
    }
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        match self {
            Connectivity::Triangular(c) => c.add_edge_adjacency_triangular(nodes_nodes),
            Connectivity::Quadrilateral(c) => c.add_edge_adjacency(nodes_nodes),
            Connectivity::Tetrahedral(c) => c.add_edge_adjacency(nodes_nodes),
            Connectivity::Hexahedral(c) => c.add_edge_adjacency(nodes_nodes),
            Connectivity::Polygonal(_) => todo!(),
            Connectivity::Polyhedral(_) => todo!(),
        }
    }
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
    pub fn element_numbers(&self) -> Option<&[usize]> {
        match self {
            Connectivity::Hexahedral(c) => c.element_numbers(),
            Connectivity::Polyhedral(c) => c.element_numbers(),
            Connectivity::Polygonal(c) => c.element_numbers(),
            Connectivity::Quadrilateral(c) => c.element_numbers(),
            Connectivity::Tetrahedral(c) => c.element_numbers(),
            Connectivity::Triangular(c) => c.element_numbers(),
        }
    }
    pub fn node_element_connectivity(&self) -> &[Vec<usize>] {
        match self {
            Connectivity::Hexahedral(c) => c.node_element_connectivity(),
            Connectivity::Polyhedral(c) => c.node_element_connectivity(),
            Connectivity::Polygonal(c) => c.node_element_connectivity(),
            Connectivity::Quadrilateral(c) => c.node_element_connectivity(),
            Connectivity::Tetrahedral(c) => c.node_element_connectivity(),
            Connectivity::Triangular(c) => c.node_element_connectivity(),
        }
    }
    pub fn number_elements(&mut self, numbers: Vec<usize>) {
        match self {
            Connectivity::Hexahedral(c) => c.number_elements(numbers),
            Connectivity::Polyhedral(c) => c.number_elements(numbers),
            Connectivity::Polygonal(c) => c.number_elements(numbers),
            Connectivity::Quadrilateral(c) => c.number_elements(numbers),
            Connectivity::Tetrahedral(c) => c.number_elements(numbers),
            Connectivity::Triangular(c) => c.number_elements(numbers),
        }
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

impl TryFrom<Connectivities> for Vec<[usize; 3]> {
    type Error = &'static str;
    fn try_from(connectivities: Connectivities) -> Result<Self, Self::Error> {
        let mut triangles = Self::new();
        for block in connectivities.into_members() {
            match block {
                Connectivity::Triangular(block) if triangles.is_empty() => {
                    triangles = block.into_iter().collect()
                }
                Connectivity::Triangular(block) => triangles.extend(block),
                _ => return Err("connectivity contains a non-triangular block"),
            }
        }
        Ok(triangles)
    }
}

impl TryFrom<Connectivities> for Vec<[usize; 8]> {
    type Error = &'static str;
    fn try_from(connectivities: Connectivities) -> Result<Self, Self::Error> {
        let mut hexes = Self::new();
        for block in connectivities.into_members() {
            match block {
                Connectivity::Hexahedral(block) if hexes.is_empty() => {
                    hexes = block.into_iter().collect()
                }
                Connectivity::Hexahedral(block) => hexes.extend(block),
                _ => return Err("connectivity contains a non-hexahedral block"),
            }
        }
        Ok(hexes)
    }
}

impl From<Vec<[usize; 3]>> for Connectivity {
    fn from(connectivity: Vec<[usize; 3]>) -> Self {
        Connectivity::Triangular(connectivity.into())
    }
}
