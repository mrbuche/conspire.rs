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

use crate::{geometry::Coordinates, math::Set};

pub struct Mesh<const D: usize> {
    connectivities: Connectivities,
    coordinates: Set<Coordinates<D>>,
}
