#[cfg(test)]
mod test;

use super::{Fitting, Peeled, merge};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::{Quantity, Tensor, TensorVec},
    units::Length,
};
use std::array::from_fn;

impl Mesh<3> {
    /// Adds a one-element-thick buffer to a hexahedral core
    /// and fits it to the target tessellation.
    ///
    /// A core boundary quadrilateral whose neighbourhood is clear of the
    /// target's corners and creases raises a single hexahedron, exactly as
    /// [`buffer`](Self::buffer) would. One that a feature crosses instead
    /// raises five pyramids sharing a new apex `m`: one on the quadrilateral
    /// `n0..n3`, one on each side quadrilateral `(n_{i+1}, n_i, m_i, m_{i+1})`
    /// spanning it to the peel duplicates. `m` is free to settle on the crease
    /// or corner, so none of the eight core nodes has to distort.
    ///
    /// The core-side face of every shell cell is the untouched quadrilateral,
    /// and every internal side quadrilateral is a whole face shared by two
    /// shell cells whichever templates meet there, so there is no diagonal to
    /// choose anywhere. Clean shell hexahedra join the core block; the pyramid
    /// fans form one pyramidal block.
    pub fn buffer_mixed(
        mut self,
        target: &Tessellation,
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        self.restrict()?;
        let boundary = self.exterior_faces();
        let Peeled {
            mut connectivities,
            mut coordinates,
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
        let mut hexes: Vec<[usize; 8]> = Vec::new();
        let mut pyramids: Vec<[usize; 5]> = Vec::new();
        boundary.iter().zip(&sizes).for_each(|(face, &size)| {
            let n: [usize; 4] = from_fn(|i| face[i]);
            let m: [usize; 4] = n.map(|node| duplicates[&node]);
            let centroid = n
                .iter()
                .map(|&node| &coordinates[node])
                .sum::<Coordinate<3>>()
                / 4.0;
            let crossed = index.nearest_corner(&centroid, size).is_some()
                || index.nearest_crease(&centroid, size).is_some();
            if crossed {
                let apex = coordinates.len();
                coordinates.push(centroid);
                layer.push(apex);
                pyramids.push([n[0], n[1], n[2], n[3], apex]);
                (0..4).for_each(|i| {
                    let j = (i + 1) % 4;
                    pyramids.push([n[j], n[i], m[i], m[j], apex]);
                });
            } else {
                hexes.push([n[0], n[1], n[2], n[3], m[0], m[1], m[2], m[3]]);
            }
        });
        if !hexes.is_empty() {
            merge(
                &mut connectivities,
                hexes,
                |connectivity| matches!(connectivity, Connectivity::Hexahedral(_)),
                Connectivity::Hexahedral,
            )?;
        }
        if !pyramids.is_empty() {
            merge(
                &mut connectivities,
                pyramids,
                |connectivity| matches!(connectivity, Connectivity::Pyramidal(_)),
                Connectivity::Pyramidal,
            )?;
        }
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

fn face_size(face: &[usize], coordinates: &Coordinates<3>) -> Quantity<Length> {
    (0..4)
        .map(|i| (&coordinates[face[(i + 1) % 4]] - &coordinates[face[i]]).norm())
        .sum::<Quantity<Length>>()
        / 4.0
}
