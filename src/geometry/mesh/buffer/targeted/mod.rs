#[cfg(test)]
mod test;

use super::{Fitting, Peeled, merge, mixed::face_size};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, PrimitiveConnectivity, Tessellation, Verdict},
    },
    math::{Quantity, Scalar, Tensor, TensorVec},
    units::Length,
};
use std::array::from_fn;

/// Shell hexahedra whose worst scaled Jacobian falls below this, and whose
/// outer face meets a feature, are replaced by pyramid fans.
const BOWTIE: Scalar = 0.15;

impl Mesh<3> {
    /// Adds a buffer layer as [`buffer`](Self::buffer) does, then replaces
    /// only the shell hexahedra that came out badly along a feature with a
    /// pyramid fan, and refits.
    pub fn buffer_targeted(
        mut self,
        target: &Tessellation,
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        self.restrict()?;
        let boundary = self.exterior_faces();
        let Peeled {
            mut connectivities,
            coordinates,
            count,
            duplicates,
            mut layer,
        } = self.peel(&boundary, 4, "non-quadrilateral boundary face")?;
        let sizes: Vec<Quantity<Length>> = boundary
            .iter()
            .map(|face| face_size(face, &coordinates))
            .collect();
        let index = target.features().index(
            sizes
                .iter()
                .copied()
                .fold(Quantity::new(0.0), Quantity::max),
        );
        let shell: Vec<[usize; 8]> = boundary
            .iter()
            .map(|face| {
                let n: [usize; 4] = from_fn(|i| face[i]);
                let m: [usize; 4] = n.map(|node| duplicates[&node]);
                [n[0], n[1], n[2], n[3], m[0], m[1], m[2], m[3]]
            })
            .collect();
        merge(
            &mut connectivities,
            shell.clone(),
            |connectivity| matches!(connectivity, Connectivity::Hexahedral(_)),
            Connectivity::Hexahedral,
        )?;
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit(&nodes, target)?;
        if let Fitting::Snap = fitting {
            mesh.project(target, &layer)?;
            mesh.fit(&(0..count).collect::<Vec<_>>(), target)?;
        }
        let block = mesh
            .connectivities()
            .iter()
            .rposition(|connectivity| matches!(connectivity, Connectivity::Hexahedral(_)))
            .ok_or("no hexahedral block")?;
        let qualities = &mesh.minimum_scaled_jacobians()[block];
        let first = qualities.len() - shell.len();
        let bad: Vec<bool> = shell
            .iter()
            .enumerate()
            .map(|(face, cell)| {
                let centroid = cell[4..]
                    .iter()
                    .map(|&node| &mesh.coordinates()[node])
                    .sum::<Coordinate<3>>()
                    / 4.0;
                qualities[first + face] < BOWTIE
                    && (index.nearest_corner(&centroid, sizes[face]).is_some()
                        || index.nearest_crease(&centroid, sizes[face]).is_some())
            })
            .collect();
        if !bad.iter().any(|&flag| flag) {
            return Ok(mesh);
        }
        let (connectivities, mut coordinates) = mesh.into();
        let mut connectivities = connectivities.into_members();
        let hexes = PrimitiveConnectivity::<3, 8>::try_from(connectivities.remove(block))?;
        let mut kept: Vec<[usize; 8]> = hexes.into_iter().collect();
        kept.truncate(first);
        let mut pyramids: Vec<[usize; 5]> = Vec::new();
        shell.iter().zip(&bad).for_each(|(cell, &flag)| {
            if flag {
                let apex = coordinates.len();
                let centroid = cell[4..]
                    .iter()
                    .map(|&node| &coordinates[node])
                    .sum::<Coordinate<3>>()
                    / 4.0;
                coordinates.push(centroid);
                layer.push(apex);
                let (n, m) = (&cell[..4], &cell[4..]);
                pyramids.push([n[0], n[1], n[2], n[3], apex]);
                (0..4).for_each(|i| {
                    let j = (i + 1) % 4;
                    pyramids.push([n[j], n[i], m[i], m[j], apex]);
                });
            } else {
                kept.push(*cell);
            }
        });
        connectivities.insert(block, Connectivity::Hexahedral(kept.into()));
        merge(
            &mut connectivities,
            pyramids,
            |connectivity| matches!(connectivity, Connectivity::Pyramidal(_)),
            Connectivity::Pyramidal,
        )?;
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit(&nodes, target)?;
        if let Fitting::Snap = fitting {
            mesh.project(target, &layer)?;
            mesh.fit(&(0..count).collect::<Vec<_>>(), target)?;
        }
        Ok(mesh)
    }
}
