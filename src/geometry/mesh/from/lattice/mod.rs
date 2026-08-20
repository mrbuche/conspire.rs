#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivities, Connectivity, Mesh},
        primitive::Solid,
    },
    math::{FxHashMap, Quantity, Scalar, Set, TensorVec},
    units::Length,
};
use std::{array::from_fn, collections::BTreeMap};

impl Mesh<3> {
    /// A uniform lattice of cells covering a solid, to be trimmed back to it.
    ///
    /// Stands a cell clear of the solid on every side, so that its surface
    /// falls strictly within the lattice and trimming finds cells to discard
    /// on either hand of it.
    pub fn lattice_over<S: Solid<3>>(solid: &S, spacing: Quantity<Length>) -> Self {
        assert!(
            spacing > Quantity::default(),
            "a lattice needs a positive spacing"
        );
        let extent = solid.bounding_box();
        let nel: [usize; 3] = from_fn(|axis| {
            ((extent.maximum()[axis] - extent.minimum()[axis]) / spacing)
                .value()
                .ceil() as usize
                + 2
        });
        let translate = Coordinate::from(from_fn::<Scalar, 3, _>(|axis| {
            (extent.minimum()[axis] - spacing).value()
        }));
        let scale = Coordinate::from([spacing.value(); 3]);
        let cells = (0..nel[2]).flat_map(move |k| {
            (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| ([i, j, k], 1)))
        });
        Self::from_lattice_cells(cells, nel, &scale, &translate)
    }
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
                coordinates.push(Coordinate::from([
                    scale[0] * (old % nxp) as f64 + translate[0],
                    scale[1] * (old / nxp % nyp) as f64 + translate[1],
                    scale[2] * (old / layer) as f64 + translate[2],
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
