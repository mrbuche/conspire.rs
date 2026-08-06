#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{FxHashMap, Set, TensorVec},
};
use std::collections::BTreeMap;

impl Mesh<3> {
    pub(crate) fn from_lattice_cells<I>(
        cells: I,
        nel: [usize; 3],
        scale: &Coordinate<3>,
        translate: &Coordinate<3>,
    ) -> Self
    where
        I: IntoIterator<Item = ([usize; 3], usize)>,
    {
        let [nx, ny, _] = nel;
        let (nxp, nyp) = (nx + 1, ny + 1);
        let layer = nxp * nyp;
        let cells = cells.into_iter();
        let (lower, _) = cells.size_hint();
        let mut connectivity = Vec::with_capacity(lower);
        let mut materials = Vec::with_capacity(lower);
        cells.for_each(|([i, j, k], material)| {
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
            materials.push(material);
        });
        let mut used: Vec<usize> = connectivity.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        let mut coordinates = Coordinates::new();
        let mapping: FxHashMap<usize, usize> = used
            .iter()
            .enumerate()
            .map(|(new, &old)| {
                coordinates.push(Coordinate::const_from([
                    (old % nxp) as f64 * scale[0] + translate[0],
                    (old / nxp % nyp) as f64 * scale[1] + translate[1],
                    (old / layer) as f64 * scale[2] + translate[2],
                ]));
                (old, new)
            })
            .collect();
        connectivity
            .iter_mut()
            .for_each(|nodes| nodes.iter_mut().for_each(|node| *node = mapping[node]));
        let mut blocks = BTreeMap::<usize, Vec<_>>::new();
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
