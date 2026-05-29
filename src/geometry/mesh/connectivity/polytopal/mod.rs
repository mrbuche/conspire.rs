use crate::geometry::mesh::connectivity::base::ConnectivityImpl;
#[cfg(feature = "netcdf")]
use crate::geometry::mesh::connectivity::base::FlatConnectivity;
use std::{fmt::Debug, num::TryFromIntError, slice, vec};

pub struct PolytopalConnectivity<const M: usize>(Vec<Vec<usize>>, Vec<Vec<usize>>);

impl<const M: usize> From<(Vec<Vec<usize>>, Vec<Vec<usize>>)> for PolytopalConnectivity<M> {
    fn from((elements_faces, faces_nodes): (Vec<Vec<usize>>, Vec<Vec<usize>>)) -> Self {
        PolytopalConnectivity(elements_faces, faces_nodes)
    }
}

impl<const M: usize> PolytopalConnectivity<M> {
    pub fn iter(&self) -> slice::Iter<'_, Vec<usize>> {
        self.0.iter()
    }
}

impl<'a, const M: usize> IntoIterator for &'a PolytopalConnectivity<M> {
    type Item = &'a Vec<usize>;
    type IntoIter = slice::Iter<'a, Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<const M: usize> IntoIterator for PolytopalConnectivity<M> {
    type Item = Vec<usize>;
    type IntoIter = vec::IntoIter<Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const M: usize> ConnectivityImpl for PolytopalConnectivity<M> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn number_of_elements(&self) -> usize {
        self.0.len()
    }
    fn number_of_faces(&self) -> Option<usize> {
        Some(self.1.len())
    }
    fn number_of_faces_per_element<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        if let Ok(num) = self.0.iter().map(|faces| faces.len().try_into()).collect() {
            Some(num)
        } else {
            panic!()
        }
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        None
    }
    fn number_of_nodes_per_face<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        if let Ok(num) = self.1.iter().map(|nodes| nodes.len().try_into()).collect() {
            Some(num)
        } else {
            panic!()
        }
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match M {
            2 => "nsided",
            3 => "nfaced",
            _ => panic!("unknown polytopal element type: M={M}"),
        }
    }
    #[cfg(feature = "netcdf")]
    fn flat_connectivity<I>(&self) -> FlatConnectivity<I>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        let elements_faces = self
            .0
            .iter()
            .flat_map(|faces| faces.iter().map(|&f| (f + 1).try_into()))
            .collect();
        let faces_nodes = self
            .1
            .iter()
            .flat_map(|nodes| nodes.iter().map(|&n| (n + 1).try_into()))
            .collect();
        match (elements_faces, faces_nodes) {
            (Ok(elements_faces), Ok(faces_nodes)) => {
                FlatConnectivity::Polytopal(elements_faces, faces_nodes)
            }
            _ => panic!(),
        }
    }
}
