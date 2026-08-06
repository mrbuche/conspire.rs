#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, grid::Voxels, mesh::Mesh};

impl Mesh<3> {
    pub fn from_voxels<T>(voxels: Voxels<T>, remove: Option<&[T]>) -> Self
    where
        T: Copy + PartialEq + Into<usize>,
    {
        Self::from_voxels_embedded(
            voxels,
            remove,
            &Coordinate::from([1.0; 3]),
            &Coordinate::from([0.0; 3]),
        )
    }
    pub(super) fn from_voxels_embedded<T>(
        voxels: Voxels<T>,
        remove: Option<&[T]>,
        scale: &Coordinate<3>,
        translate: &Coordinate<3>,
    ) -> Self
    where
        T: Copy + PartialEq + Into<usize>,
    {
        let nel = *voxels.nel();
        Self::from_lattice_cells(
            voxels
                .logical_iter()
                .filter(|&(_index, &block)| remove.is_none_or(|ids| !ids.contains(&block)))
                .map(|(index, &block)| (index, block.into())),
            nel,
            scale,
            translate,
        )
    }
}
