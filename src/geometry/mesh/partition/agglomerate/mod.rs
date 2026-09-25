#[cfg(test)]
mod test;

use super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{Set, Tensor, TensorVec},
};
use std::collections::HashMap;

impl Partition {
    pub fn agglomerate(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        let elements_faces = outward_faces(mesh)?;
        let mut faces_nodes: Vec<Vec<usize>> = Vec::new();
        let mut indices: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut cells = Vec::new();
        for part in 0..self.number_of_parts() {
            let elements = self.part_elements(part);
            if elements.is_empty() {
                continue;
            }
            let mut roots = (0..elements.len()).collect::<Vec<_>>();
            let mut tally: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
            let mut order: Vec<(&Vec<usize>, Vec<usize>)> = Vec::new();
            for (position, &element) in elements.iter().enumerate() {
                for face in &elements_faces[element] {
                    let mut key = face.clone();
                    key.sort_unstable();
                    match tally.get_mut(&key) {
                        Some((count, first)) => {
                            *count += 1;
                            let (a, b) = (find(&mut roots, *first), find(&mut roots, position));
                            roots[b] = a;
                        }
                        None => {
                            tally.insert(key.clone(), (1, position));
                            order.push((face, key))
                        }
                    }
                }
            }
            let root = find(&mut roots, 0);
            if (1..elements.len()).any(|position| find(&mut roots, position) != root) {
                return Err("a part is not connected through shared faces");
            }
            cells.push(
                order
                    .into_iter()
                    .filter(|(_, key)| tally[key].0 == 1)
                    .map(|(face, key)| {
                        *indices.entry(key).or_insert_with(|| {
                            faces_nodes.push(face.clone());
                            faces_nodes.len() - 1
                        })
                    })
                    .collect::<Vec<usize>>(),
            );
        }
        let mut remap = vec![usize::MAX; mesh.number_of_nodes()];
        faces_nodes
            .iter()
            .flatten()
            .for_each(|&node| remap[node] = 0);
        let mut coordinates = Coordinates::new();
        remap.iter_mut().enumerate().for_each(|(node, new)| {
            if *new == 0 {
                *new = coordinates.len();
                coordinates.push(mesh.coordinates()[node].clone())
            }
        });
        faces_nodes
            .iter_mut()
            .flatten()
            .for_each(|node| *node = remap[*node]);
        Ok((
            Connectivities::from(vec![Connectivity::Polyhedral((cells, faces_nodes).into())]),
            Set::from(coordinates),
        )
            .into())
    }
}

fn find(roots: &mut [usize], mut node: usize) -> usize {
    while roots[node] != node {
        roots[node] = roots[roots[node]];
        node = roots[node]
    }
    node
}

fn outward_faces(mesh: &Mesh<3>) -> Result<Vec<Vec<Vec<usize>>>, &'static str> {
    let mut elements_faces = Vec::with_capacity(mesh.number_of_elements());
    for block in mesh.iter() {
        match block {
            Connectivity::Polyhedral(connectivity) => {
                let mut owner = HashMap::new();
                connectivity
                    .elements_faces()
                    .iter()
                    .enumerate()
                    .for_each(|(element, faces)| {
                        faces.iter().for_each(|&face| {
                            owner.entry(face).or_insert(element);
                        })
                    });
                connectivity
                    .elements_faces()
                    .iter()
                    .enumerate()
                    .for_each(|(element, faces)| {
                        elements_faces.push(
                            faces
                                .iter()
                                .map(|&face| {
                                    let mut nodes = connectivity.faces_nodes()[face].clone();
                                    if owner[&face] != element {
                                        nodes.reverse()
                                    }
                                    nodes
                                })
                                .collect(),
                        )
                    })
            }
            Connectivity::Polygonal(_)
            | Connectivity::Quadrilateral(_)
            | Connectivity::Triangular(_) => {
                return Err("agglomeration requires three-dimensional elements");
            }
            _ => block
                .iter()
                .for_each(|element| elements_faces.push(block.element_faces(element))),
        }
    }
    Ok(elements_faces)
}
