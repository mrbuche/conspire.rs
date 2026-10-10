pub mod conduction;

use crate::vem::block::{Block, element::thermal::ElementNodalTemperatures};

pub use crate::domain::thermal::NodalTemperatures;

impl<C, F, R> Block<C, F, R> {
    pub(crate) fn element_temperatures(
        nodes: &[usize],
        nodal_temperatures: &NodalTemperatures,
    ) -> ElementNodalTemperatures {
        nodes.iter().map(|&node| nodal_temperatures[node]).collect()
    }
}
