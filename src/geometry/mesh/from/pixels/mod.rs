#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Pixels,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{Set, Tensor, TensorVec},
};
use std::collections::BTreeMap;

impl Mesh<2> {
    pub fn from_pixels<T>(pixels: Pixels<T>, remove: Option<&[T]>) -> Self
    where
        T: Copy + PartialEq + Into<usize>,
    {
        let [nx, ny] = *pixels.nel();
        let (nxp, nyp) = (nx + 1, ny + 1);
        let nodes_unfiltered = nxp * nyp;
        let mut connectivity = Vec::with_capacity(pixels.len());
        let mut materials = Vec::with_capacity(pixels.len());
        let (mut i, mut j) = (0, 0);
        for &block in pixels.data() {
            if remove.is_none_or(|ids| !ids.contains(&block)) {
                let base = i + nxp * j;
                connectivity.push([base, base + 1, base + nxp + 1, base + nxp]);
                materials.push(block.into());
            }
            i += 1;
            if i == nx {
                i = 0;
                j += 1;
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
                let y = old / nxp;
                coordinates.push(Coordinate::const_from([x as f64, y as f64]));
            }
        }
        connectivity
            .iter_mut()
            .for_each(|nodes| nodes.iter_mut().for_each(|node| *node = mapping[*node]));
        let mut blocks = BTreeMap::<usize, Vec<_>>::new();
        connectivity
            .into_iter()
            .zip(materials)
            .for_each(|(nodes, material)| blocks.entry(material).or_default().push(nodes));
        let mut connectivities = Vec::with_capacity(blocks.len());
        let mut numbers = Vec::with_capacity(blocks.len());
        for (material, quads) in blocks {
            numbers.push(material);
            connectivities.push(Connectivity::Quadrilateral(quads.into()));
        }
        (
            Connectivities::from((connectivities, numbers)),
            Set::from(coordinates),
        )
            .into()
    }
}
