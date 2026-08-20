#[cfg(test)]
mod test;

mod fit;
mod restrict;

use super::{Connectivity, Mesh};
use crate::{
    geometry::primitive::Solid,
    math::{Tensor, TensorVec},
};
use std::collections::{HashMap, hash_map::Entry};

/// Constraint on how the buffer layer approaches the target surface.
#[derive(Clone, Copy, Debug)]
pub enum Fitting {
    /// The layer settles wherever the quality and fit energies balance.
    Soft,
    /// The layer settles as above, but is then projected onto the surface,
    /// after which the interior relaxes.
    Snap,
}

impl Mesh<3> {
    pub fn buffer<S: Solid<3>>(
        mut self,
        target: &S,
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        if target.is_empty() {
            return Err("empty tessellation");
        }
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
        let cells = boundary
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
            .collect::<Vec<_>>();
        match connectivities
            .iter()
            .rposition(|connectivity| matches!(connectivity, Connectivity::Hexahedral(_)))
        {
            Some(index) => {
                let Connectivity::Hexahedral(hexes) = connectivities.remove(index) else {
                    unreachable!()
                };
                connectivities.insert(
                    index,
                    Connectivity::Hexahedral(
                        hexes.into_iter().chain(cells).collect::<Vec<_>>().into(),
                    ),
                );
            }
            None => connectivities.push(Connectivity::Hexahedral(cells.into())),
        }
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit(&nodes, target)?;
        if let Fitting::Snap = fitting {
            let coordinates = mesh.coordinates.members_mut();
            layer
                .iter()
                .for_each(|&node| coordinates[node] = target.closest_point(&coordinates[node]).0);
            mesh.fit(&(0..count).collect::<Vec<_>>(), target)?;
        }
        Ok(mesh)
    }
}
