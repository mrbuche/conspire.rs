#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinate,
    mesh::Mesh,
    solid::{Partition, SolidOracle},
};
use std::array::from_fn;

impl Mesh<3> {
    /// Hexahedra on a uniform lattice between `lower` and `upper`, one block per volume of the
    /// partition, labeled by its index.
    ///
    /// A cell belongs to the volume owning its center and is dropped where no volume does.
    /// Neighbouring blocks share the nodes of the faces between them, so the mesh conforms to
    /// every interface as a staircase.
    pub fn from_partition<O: SolidOracle>(
        partition: &Partition<O>,
        lower: &Coordinate<3>,
        upper: &Coordinate<3>,
        nel: [usize; 3],
    ) -> Result<Self, &'static str> {
        if nel.contains(&0) {
            return Err("the lattice needs at least one cell along each axis");
        }
        let lower: [f64; 3] = from_fn(|a| lower[a].value());
        let size: [f64; 3] = from_fn(|a| (upper[a].value() - lower[a]) / nel[a] as f64);
        if size.iter().any(|&h| h <= 0.0) {
            return Err("the upper corner must be above the lower one along each axis");
        }
        let cells = (0..nel[2])
            .flat_map(|k| (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| [i, j, k])));
        Ok(Self::from_lattice_cells(
            cells.filter_map(|cell| {
                let center =
                    Coordinate::from(from_fn(|a| lower[a] + size[a] * (cell[a] as f64 + 0.5)));
                partition.owner(&center).map(|volume| (cell, volume))
            }),
            nel,
            &Coordinate::from(size),
            &Coordinate::from(lower),
        ))
    }
}
