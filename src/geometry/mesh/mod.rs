mod base;
mod connectivity;
pub(crate) mod from;
mod into;
mod read;
mod tessellation;
mod write;

pub use self::{
    connectivity::{Connectivities, Connectivity},
    read::Input,
    tessellation::Tessellation,
    write::Output,
};

use crate::geometry::Coordinates;

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Coordinates<D>,
}
