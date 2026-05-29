use crate::geometry::mesh::connectivity::base::ConnectivityImpl;
use std::slice::Iter;

pub struct PrimitiveConnectivity<const M: usize, const N: usize>(Vec<[usize; N]>);

impl<const M: usize, const N: usize> From<Vec<[usize; N]>> for PrimitiveConnectivity<M, N> {
    fn from(connectivity: Vec<[usize; N]>) -> Self {
        PrimitiveConnectivity(connectivity)
    }
}

impl<const M: usize, const N: usize> PrimitiveConnectivity<M, N> {
    pub fn iter(&self) -> Iter<'_, [usize; N]> {
        self.0.iter()
    }
}

impl<'a, const M: usize, const N: usize> IntoIterator for &'a PrimitiveConnectivity<M, N> {
    type Item = &'a [usize; N];
    type IntoIter = Iter<'a, [usize; N]>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<const M: usize, const N: usize> IntoIterator for PrimitiveConnectivity<M, N> {
    type Item = [usize; N];
    type IntoIter = std::vec::IntoIter<[usize; N]>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const M: usize, const N: usize> ConnectivityImpl for PrimitiveConnectivity<M, N> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        Some(N)
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match (M, N) {
            (2, 3) => "tri3",
            (2, 4) => "quad4",
            (3, 4) => "tet4",
            (3, 8) => "hex8",
            _ => panic!("unknown primitive element type: M={M}, N={N}"),
        }
    }
    #[cfg(feature = "netcdf")]
    fn primitive_connectivity_flattened(&self) -> Option<Vec<i32>> {
        Some(
            self.0
                .iter()
                .flat_map(|nodes| nodes.iter().map(|&node| node as i32 + 1))
                .collect(),
        )
    }
}
