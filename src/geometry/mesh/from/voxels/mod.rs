#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{Set, Tensor, TensorVec},
};
use std::collections::BTreeMap;

impl Mesh<3> {
    pub fn from_voxels<T>(voxels: Voxels<T>, remove: Option<&[T]>) -> Self
    where
        T: Copy + PartialEq + Into<usize>,
    {
        let [nx, ny, nz] = *voxels.nel();
        let (nxp, nyp, nzp) = (nx + 1, ny + 1, nz + 1);
        let layer = nxp * nyp;
        let nodes_unfiltered = layer * nzp;
        let mut connectivity = Vec::with_capacity(voxels.len());
        let mut materials = Vec::with_capacity(voxels.len());
        let (mut i, mut j, mut k) = (0, 0, 0);
        for &block in voxels.data() {
            if remove.is_none_or(|ids| !ids.contains(&block)) {
                let base = i + nxp * j + layer * k;
                let top = base + layer;
                connectivity.push([
                    base,
                    base + 1,
                    base + nxp + 1,
                    base + nxp,
                    top,
                    top + 1,
                    top + nxp + 1,
                    top + nxp,
                ]);
                materials.push(block.into());
            }
            i += 1;
            if i == nx {
                i = 0;
                j += 1;
                if j == ny {
                    j = 0;
                    k += 1;
                }
            }
        }
        let mut used = vec![false; nodes_unfiltered];
        connectivity
            .iter()
            .for_each(|nodes| nodes.iter().for_each(|&node| used[node] = true));
        let mut mapping = vec![0usize; nodes_unfiltered];
        let mut coordinates = Coordinates::new();
        for (old, &is_used) in used.iter().enumerate() {
            if is_used {
                mapping[old] = coordinates.len();
                let x = old % nxp;
                let y = old / nxp % nyp;
                let z = old / layer;
                coordinates.push(Coordinate::const_from([x as f64, y as f64, z as f64]));
            }
        }
        connectivity
            .iter_mut()
            .for_each(|nodes| nodes.iter_mut().for_each(|node| *node = mapping[*node]));
        let mut blocks: BTreeMap<usize, Vec<[usize; 8]>> = BTreeMap::new();
        connectivity
            .into_iter()
            .zip(materials)
            .for_each(|(nodes, material)| blocks.entry(material).or_default().push(nodes));
        let mut connectivities = Vec::with_capacity(blocks.len());
        let mut numbers = Vec::with_capacity(blocks.len());
        for (material, hexes) in blocks {
            numbers.push(material);
            connectivities.push(Connectivity::Hexahedral(hexes.into()));
        }
        (
            Connectivities::from((connectivities, numbers)),
            Set::from(coordinates),
        )
            .into()
    }
}
