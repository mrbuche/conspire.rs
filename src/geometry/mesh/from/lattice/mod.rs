#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivities, Connectivity, Mesh,
            from::{kuhn, orient},
        },
    },
    math::{FxHashMap, Set, TensorVec},
};
use std::{array::from_fn, collections::BTreeMap};

fn remap<const N: usize>(
    connectivity: &mut [[usize; N]],
    nel: [usize; 3],
    scale: &Coordinate<3>,
    translate: &Coordinate<3>,
) -> Coordinates<3> {
    let [nx, ny, _] = nel;
    let (nxp, nyp) = (nx + 1, ny + 1);
    let layer = nxp * nyp;
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
    coordinates
}

fn blocks<const N: usize>(
    connectivity: Vec<[usize; N]>,
    materials: Vec<usize>,
    variant: impl Fn(Vec<[usize; N]>) -> Connectivity,
) -> Connectivities {
    let mut grouped = BTreeMap::<usize, Vec<_>>::new();
    connectivity
        .into_iter()
        .zip(materials)
        .for_each(|(nodes, material)| grouped.entry(material).or_default().push(nodes));
    let mut connectivities = Vec::with_capacity(grouped.len());
    let mut numbers = Vec::with_capacity(grouped.len());
    for (material, elements) in grouped {
        numbers.push(material);
        connectivities.push(variant(elements));
    }
    Connectivities::from((connectivities, numbers))
}

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
        let coordinates = remap(&mut connectivity, nel, scale, translate);
        (
            blocks(connectivity, materials, |elements| {
                Connectivity::Hexahedral(elements.into())
            }),
            Set::from(coordinates),
        )
            .into()
    }
    /// Meshes lattice cells as six tetrahedra apiece, by the Kuhn/Freudenthal
    /// split about the diagonal from each cell's lowest-numbered corner to its
    /// highest.
    ///
    /// Node numbering increases with each coordinate, so those two corners are
    /// the lexicographic extremes of the cell, and each square face is cut by
    /// the diagonal joining its own two extremes. Neighboring cells share a
    /// face's extremes, so they cut it the same way and the mesh is conforming
    /// without any parity rule.
    #[allow(dead_code)]
    pub(crate) fn from_lattice_tets<I>(
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
        let steps = [1, nxp, layer];
        let cells = cells.into_iter();
        let (lower, _) = cells.size_hint();
        let mut connectivity = Vec::with_capacity(6 * lower);
        let mut materials = Vec::with_capacity(6 * lower);
        cells.for_each(|([i, j, k], material)| {
            let low = i + nxp * j + layer * k;
            let corners: [usize; 8] = from_fn(|corner| {
                low + (0..3)
                    .map(|axis| ((corner >> axis) & 1) * steps[axis])
                    .sum::<usize>()
            });
            kuhn(&corners).into_iter().for_each(|tet| {
                connectivity.push(tet);
                materials.push(material);
            })
        });
        let coordinates = remap(&mut connectivity, nel, scale, translate);
        orient(&mut connectivity, &coordinates);
        (
            blocks(connectivity, materials, |elements| {
                Connectivity::Tetrahedral(elements.into())
            }),
            Set::from(coordinates),
        )
            .into()
    }
}
