#[cfg(feature = "netcdf")]
use crate::geometry::mesh::connectivity::base::FlatConnectivity;
use crate::{geometry::mesh::connectivity::base::ConnectivityImpl, math::Sets};
use std::{fmt::Debug, num::TryFromIntError, slice::Iter, vec::IntoIter};

pub struct PrimitiveConnectivity<const M: usize, const N: usize>(Sets<Vec<[usize; N]>>);

impl<const M: usize, const N: usize> From<Vec<[usize; N]>> for PrimitiveConnectivity<M, N> {
    fn from(connectivity: Vec<[usize; N]>) -> Self {
        PrimitiveConnectivity(Sets::from(connectivity))
    }
}

impl<const M: usize, const N: usize> PrimitiveConnectivity<M, N> {
    pub fn iter(&self) -> Iter<'_, [usize; N]> {
        self.0.members().iter()
    }
}

impl<'a, const M: usize, const N: usize> IntoIterator for &'a PrimitiveConnectivity<M, N> {
    type Item = &'a [usize; N];
    type IntoIter = Iter<'a, [usize; N]>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.members().iter()
    }
}

impl<const M: usize, const N: usize> IntoIterator for PrimitiveConnectivity<M, N> {
    type Item = [usize; N];
    type IntoIter = IntoIter<[usize; N]>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_members().into_iter()
    }
}

impl<const M: usize, const N: usize> ConnectivityImpl for PrimitiveConnectivity<M, N> {
    fn is_empty(&self) -> bool {
        self.0.members().is_empty()
    }
    fn element_numbers(&self) -> Option<&[usize]> {
        self.0.numbers()
    }
    fn node_element_connectivity(&self) -> &[Vec<usize>] {
        self.0.converse()
    }
    fn number_elements(&mut self, numbers: Vec<usize>) {
        self.0.set_numbers(numbers)
    }
    fn number_of_elements(&self) -> usize {
        self.0.members().len()
    }
    fn number_of_faces(&self) -> Option<usize> {
        None
    }
    fn number_of_faces_per_element<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        None
    }
    fn number_of_nodes_per_element(&self) -> Option<usize> {
        Some(N)
    }
    fn number_of_nodes_per_face<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        None
    }
    #[cfg(feature = "netcdf")]
    fn exodus_element_type(&self) -> &str {
        match (M, N) {
            (2, 3) => "tri3",
            (2, 4) => "quad4",
            (3, 4) => "tet4",
            (3, 5) => "pyramid5",
            (3, 6) => "wedge6",
            (3, 8) => "hex8",
            _ => panic!("unknown primitive element type: M={M}, N={N}"),
        }
    }
    #[cfg(feature = "netcdf")]
    fn flat_connectivity<I>(&self) -> FlatConnectivity<I>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        match self
            .0
            .members()
            .iter()
            .flat_map(|nodes| nodes.iter().map(|&node| (node + 1).try_into()))
            .collect()
        {
            Ok(flat) => FlatConnectivity::Primitive(flat),
            Err(_) => panic!(),
        }
    }
}
impl PrimitiveConnectivity<2, 3> {
    pub fn add_edge_adjacency_triangular(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[a, b, c] in self.0.members() {
            nodes_nodes[a].push(b);
            nodes_nodes[b].push(a);
            nodes_nodes[b].push(c);
            nodes_nodes[c].push(b);
            nodes_nodes[c].push(a);
            nodes_nodes[a].push(c);
        }
    }
}

impl PrimitiveConnectivity<2, 4> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[a, b, c, d] in self.0.members() {
            nodes_nodes[a].push(b);
            nodes_nodes[b].push(a);
            nodes_nodes[b].push(c);
            nodes_nodes[c].push(b);
            nodes_nodes[c].push(d);
            nodes_nodes[d].push(c);
            nodes_nodes[d].push(a);
            nodes_nodes[a].push(d);
        }
    }
}

impl PrimitiveConnectivity<3, 4> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[a, b, c, d] in self.0.members() {
            for (u, v) in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
                nodes_nodes[u].push(v);
                nodes_nodes[v].push(u);
            }
        }
    }
}

impl PrimitiveConnectivity<3, 5> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[a, b, c, d, e] in self.0.members() {
            for (u, v) in [
                (a, b),
                (b, c),
                (c, d),
                (d, a),
                (a, e),
                (b, e),
                (c, e),
                (d, e),
            ] {
                nodes_nodes[u].push(v);
                nodes_nodes[v].push(u);
            }
        }
    }
}

impl PrimitiveConnectivity<3, 6> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[a, b, c, d, e, f] in self.0.members() {
            for (u, v) in [
                (a, b),
                (b, c),
                (c, a),
                (d, e),
                (e, f),
                (f, d),
                (a, d),
                (b, e),
                (c, f),
            ] {
                nodes_nodes[u].push(v);
                nodes_nodes[v].push(u);
            }
        }
    }
}

impl PrimitiveConnectivity<3, 8> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for &[n0, n1, n2, n3, n4, n5, n6, n7] in self.0.members() {
            for (u, v) in [
                (n0, n1),
                (n1, n2),
                (n2, n3),
                (n3, n0),
                (n4, n5),
                (n5, n6),
                (n6, n7),
                (n7, n4),
                (n0, n4),
                (n1, n5),
                (n2, n6),
                (n3, n7),
            ] {
                nodes_nodes[u].push(v);
                nodes_nodes[v].push(u);
            }
        }
    }
}
