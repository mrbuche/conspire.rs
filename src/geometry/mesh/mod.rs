mod base;
mod connectivity;
pub(crate) mod from;
mod into;
mod read;
mod tessellation;
mod write;

pub use self::{
    connectivity::{Connectivities, Connectivity},
    tessellation::Tessellation,
};

#[cfg(feature = "netcdf")]
pub use self::{read::ReadExodus, write::exodus::WriteExodus};

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Coordinates<D>,
}
