use crate::geometry::mesh::Connectivity;
use std::collections::HashSet;

pub(super) fn element_faces(block: &Connectivity, element: &[usize]) -> Vec<Vec<usize>> {
    match block {
        Connectivity::Polyhedral(connectivity) => element
            .iter()
            .map(|&face| connectivity.faces_nodes()[face].clone())
            .collect(),
        Connectivity::Polygonal(connectivity) => element
            .iter()
            .map(|&face| connectivity.faces_nodes()[face].clone())
            .collect(),
        _ => block
            .local_faces()
            .iter()
            .map(|face| face.iter().map(|&local| element[local]).collect())
            .collect(),
    }
}

pub(super) fn element_edges(faces: &[Vec<usize>]) -> Vec<[usize; 2]> {
    let mut edges = HashSet::new();
    faces.iter().for_each(|face| {
        let n = face.len();
        (0..n).for_each(|i| {
            let mut key = [face[i], face[(i + 1) % n]];
            key.sort_unstable();
            edges.insert(key);
        });
    });
    edges.into_iter().collect()
}
