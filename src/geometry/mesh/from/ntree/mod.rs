#[cfg(test)]
mod test;

mod dualization;
pub(crate) mod facets;
mod polyhedra;
mod tetrahedra;

pub use dualization::Dualization;
#[cfg(test)]
pub(crate) use dualization::verify_dual;
pub(crate) use dualization::{leaf_containing, leaf_containing_from};
