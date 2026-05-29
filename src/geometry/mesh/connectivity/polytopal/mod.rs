use crate::geometry::mesh::connectivity::base::ConnectivityImpl;
use std::{slice, vec};

pub struct PolytopalConnectivity<const M: usize>(Vec<Vec<usize>>);

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
    fn len(&self) -> usize {
        self.0.len()
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        None
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
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        None
    }
}
