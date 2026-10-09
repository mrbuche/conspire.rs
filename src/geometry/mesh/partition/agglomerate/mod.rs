#[cfg(test)]
mod test;

use super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{Boundary, Connectivities, Connectivity, Mesh},
    },
    math::{Set, Tensor, TensorVec},
};
use std::collections::HashMap;

impl Partition {
    pub fn agglomerate(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        let boundary = Boundary::try_from(mesh)?;
        let mut faces_nodes = Vec::<Vec<usize>>::new();
        let mut indices = HashMap::<Vec<usize>, usize>::new();
        let mut cells = Vec::new();
        for part in 0..self.number_of_parts() {
            let elements = self.part_elements(part);
            if elements.is_empty() {
                continue;
            }
            cells.push(
                boundary
                    .faces(elements)?
                    .into_iter()
                    .map(|face| {
                        let mut key = face.clone();
                        key.sort_unstable();
                        *indices.entry(key).or_insert_with(|| {
                            faces_nodes.push(face);
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
