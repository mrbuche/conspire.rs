#[cfg(test)]
mod test;

use super::{Fitting, Peeled, merge};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::{Tensor, TensorVec},
};
use std::array::from_fn;

impl Mesh<3> {
    /// Adds a one-element-thick shell of pyramids to a hexahedral core and fits
    /// it to the target.
    ///
    /// Every core boundary quadrilateral `n0..n3` raises five pyramids sharing
    /// a new apex `m` above it: one on the quadrilateral itself and one on each
    /// side quadrilateral `(n_{i+1}, n_i, m_i, m_{i+1})` spanning it to its
    /// peel duplicates. The core-side face of every shell cell is the untouched
    /// quadrilateral, so the core stays a single hexahedral block and the shell
    /// is a single pyramidal block. Every internal side quadrilateral is a
    /// whole face shared by two shell cells, so there is no diagonal to choose.
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
        let mut pyramids: Vec<[usize; 5]> = Vec::with_capacity(boundary.len() * 5);
        boundary.iter().for_each(|face| {
            let n: [usize; 4] = from_fn(|i| face[i]);
            let m: [usize; 4] = n.map(|node| duplicates[&node]);
            let apex = coordinates.len();
            let centroid = n
                .iter()
                .map(|&node| &coordinates[node])
                .sum::<Coordinate<3>>()
                / 4.0;
            coordinates.push(centroid);
            layer.push(apex);
            pyramids.push([n[0], n[1], n[2], n[3], apex]);
            (0..4).for_each(|i| {
                let j = (i + 1) % 4;
                pyramids.push([n[j], n[i], m[i], m[j], apex]);
            });
        });
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
