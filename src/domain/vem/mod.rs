//! Virtual element library.

pub mod block;
mod from;

pub type NodalCoordinates = crate::fem::NodalCoordinates<3>;
pub type NodalReferenceCoordinates = crate::fem::NodalReferenceCoordinates<3>;
