mod graph;

use crate::geometry::mesh::connectivity::base::FlatConnectivity;
use crate::{geometry::mesh::connectivity::base::ConnectivityImpl, math::Labels};
use std::{fmt::Debug, num::TryFromIntError, slice::Iter, vec::IntoIter};

pub struct PolytopalConnectivity<const M: usize>(Vec<Vec<usize>>, Vec<Vec<usize>>, Option<Labels>);

impl<const M: usize> From<(Vec<Vec<usize>>, Vec<Vec<usize>>)> for PolytopalConnectivity<M> {
    fn from((elements_faces, faces_nodes): (Vec<Vec<usize>>, Vec<Vec<usize>>)) -> Self {
        PolytopalConnectivity(elements_faces, faces_nodes, None)
    }
}

impl<const M: usize> PolytopalConnectivity<M> {
    pub fn add_edge_adjacency(&self, nodes_nodes: &mut [Vec<usize>]) {
        for faces in &self.0 {
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
        &self.0
    }
    pub fn faces_nodes(&self) -> &[Vec<usize>] {
        &self.1
    }
    pub fn iter(&self) -> Iter<'_, Vec<usize>> {
        self.0.iter()
    }
}

impl<'a, const M: usize> IntoIterator for &'a PolytopalConnectivity<M> {
    type Item = &'a Vec<usize>;
    type IntoIter = Iter<'a, Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<const M: usize> IntoIterator for PolytopalConnectivity<M> {
    type Item = Vec<usize>;
    type IntoIter = IntoIter<Vec<usize>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const M: usize> ConnectivityImpl for PolytopalConnectivity<M> {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn element_labels(&self) -> Option<&[usize]> {
        self.2.as_ref().map(Labels::as_slice)
    }
    fn label_elements(&mut self, labels: Vec<usize>) {
        self.2 = Some(Labels::new(labels, self.0.len()))
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
