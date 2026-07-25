use crate::geometry::mesh::Connectivity;
use std::collections::HashSet;

pub(super) fn element_faces(block: &Connectivity, element: &[usize]) -> Vec<Vec<usize>> {
    block.element_faces(element)
}

pub(super) fn face_owners(block: &Connectivity) -> Option<Vec<usize>> {
    let elements_faces = match block {
        Connectivity::Polyhedral(connectivity) => connectivity.elements_faces(),
        Connectivity::Polygonal(connectivity) => connectivity.elements_faces(),
        _ => return None,
    };
    let number_of_faces = elements_faces.iter().flatten().max().map_or(0, |&f| f + 1);
    let mut owners = vec![usize::MAX; number_of_faces];
    elements_faces
        .iter()
        .enumerate()
        .for_each(|(element, faces)| {
            faces
                .iter()
                .for_each(|&face| owners[face] = owners[face].min(element))
        });
    Some(owners)
}

pub(super) fn oriented_element_faces(
    block: &Connectivity,
    element: &[usize],
    local: usize,
    owners: Option<&[usize]>,
) -> Vec<Vec<usize>> {
    let faces = element_faces(block, element);
    match owners {
        Some(owners) => element
            .iter()
            .zip(faces)
            .map(|(&face, polygon)| {
                if owners[face] == local {
                    polygon
                } else {
                    polygon.into_iter().rev().collect()
                }
            })
            .collect(),
        None => faces,
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
