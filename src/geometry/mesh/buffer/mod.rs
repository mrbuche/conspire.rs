#[cfg(test)]
mod test;

mod fit;
mod restrict;

use super::{Connectivity, Mesh, Tessellation};
use crate::math::{Tensor, TensorVec};
use std::collections::{HashMap, hash_map::Entry};

impl Mesh<3> {
    pub fn buffer(mut self, target: &Tessellation) -> Result<Self, &'static str> {
        self.restrict()?;
        let boundary = self.exterior_faces();
        let mut edges = HashMap::new();
        boundary.iter().try_for_each(|face| {
            if face.len() != 4 {
                return Err("non-quadrilateral boundary face");
            }
            (0..4).for_each(|i| {
                let mut edge = [face[i], face[(i + 1) % 4]];
                edge.sort_unstable();
                *edges.entry(edge).or_insert(0u8) += 1;
            });
            Ok(())
        })?;
        if edges.values().any(|&count| count != 2) {
            return Err("non-manifold boundary");
        }
        let (connectivities, mut coordinates) = self.into();
        let mut connectivities = connectivities.into_members();
        let count = coordinates.len();
        let mut duplicates = HashMap::new();
        let mut layer = Vec::new();
        boundary.iter().flatten().for_each(|&node| {
            if let Entry::Vacant(slot) = duplicates.entry(node) {
                slot.insert(coordinates.len());
                layer.push(coordinates.len());
                let point = coordinates[node].clone();
                coordinates.push(point);
            }
        });
        connectivities.push(Connectivity::Hexahedral(
            boundary
                .iter()
                .map(|face| {
                    [
                        face[0],
                        face[1],
                        face[2],
                        face[3],
                        duplicates[&face[0]],
                        duplicates[&face[1]],
                        duplicates[&face[2]],
                        duplicates[&face[3]],
                    ]
                })
                .collect::<Vec<_>>()
                .into(),
        ));
        let mut mesh = Self::from((connectivities, coordinates));
        let mut nodes = layer;
        nodes.extend(0..count);
        mesh.fit(&nodes, target)?;
        Ok(mesh)
    }
}
