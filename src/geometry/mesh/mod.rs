mod base;
mod connectivity;
pub(crate) mod from;
mod into;
mod tessellation;
mod write;

#[cfg(feature = "netcdf")]
mod exodus;

pub use self::{
    connectivity::{Connectivities, Connectivity},
    tessellation::Tessellation,
};

#[cfg(feature = "netcdf")]
pub use self::exodus::WriteExodus;

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Coordinates<D>,
}
