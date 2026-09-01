use crate::geometry::mesh::connectivity::base::FlatConnectivity;
use crate::{geometry::mesh::connectivity::base::ConnectivityImpl, math::Set};
use std::{cell::OnceCell, fmt::Debug, num::TryFromIntError, slice::Iter, vec::IntoIter};

pub struct PolytopalConnectivity<const M: usize>(
    Set<Vec<Vec<usize>>>,
    Vec<Vec<usize>>,
    OnceCell<Vec<Vec<usize>>>,
);

impl<const M: usize> From<(Vec<Vec<usize>>, Vec<Vec<usize>>)> for PolytopalConnectivity<M> {
    fn from((elements_faces, faces_nodes): (Vec<Vec<usize>>, Vec<Vec<usize>>)) -> Self {
        PolytopalConnectivity(Set::from(elements_faces), faces_nodes, OnceCell::new())
    }
}

impl<const M: usize> PolytopalConnectivity<M> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for faces in self.0.members() {
            for &face in faces {
                let nodes = &self.1[face];
                for (index, &u) in nodes.iter().enumerate() {
                    let v = nodes[(index + 1) % nodes.len()];
                    if u != v {
                        nodes_nodes[u].push(v);
                        nodes_nodes[v].push(u);
                    }
                }
            }
        }
    }
    pub fn elements_faces(&self) -> &[Vec<usize>] {
        self.0.members()
    }
    pub fn faces_nodes(&self) -> &[Vec<usize>] {
        &self.1
    }
    pub fn iter(&self) -> Iter<'_, Vec<usize>> {
        self.0.members().iter()
    }
}

impl<'a, const M: usize> IntoIterator for &'a PolytopalConnectivity<M> {
    type Item = &'a Vec<usize>;
    type IntoIter = Iter<'a, Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.members().iter()
    }
}

impl<const M: usize> IntoIterator for PolytopalConnectivity<M> {
    type Item = Vec<usize>;
    type IntoIter = IntoIter<Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_members().into_iter()
    }
}

impl<const M: usize> ConnectivityImpl for PolytopalConnectivity<M> {
    fn is_empty(&self) -> bool {
        self.0.members().is_empty()
    }
    fn element_numbers(&self) -> Option<&[usize]> {
        self.0.numbers()
    }
    fn node_element_connectivity(&self) -> &[Vec<usize>] {
        self.2.get_or_init(|| {
            let number_of_nodes = self.1.iter().flatten().max().map_or(0, |&node| node + 1);
            let mut nodes_elements = vec![Vec::new(); number_of_nodes];
            for (element, faces) in self.0.members().iter().enumerate() {
                let mut nodes: Vec<usize> = faces
                    .iter()
                    .flat_map(|&face| self.1[face].iter().copied())
                    .collect();
                nodes.sort_unstable();
                nodes.dedup();
                for node in nodes {
                    nodes_elements[node].push(element)
                }
            }
            nodes_elements
        })
    }
    fn number_elements(&mut self, numbers: Vec<usize>) {
        self.0.set_numbers(numbers)
    }
    fn number_of_elements(&self) -> usize {
        self.0.members().len()
    }
    fn number_of_faces(&self) -> Option<usize> {
        Some(self.1.len())
    }
    fn number_of_faces_per_element<I>(&self) -> Option<Vec<I>>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        if let Ok(num) = self
            .0
            .members()
            .iter()
            .map(|faces| faces.len().try_into())
            .collect()
        {
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
    fn exodus_element_type(&self) -> &str {
        match M {
            2 => "nsided",
            3 => "nfaced",
            _ => panic!(),
        }
    }
    fn flat_connectivity<I>(&self) -> FlatConnectivity<I>
    where
        I: Debug + TryFrom<usize, Error = TryFromIntError>,
    {
        let elements_faces = self
            .0
            .members()
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
