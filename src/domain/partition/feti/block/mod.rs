#![allow(dead_code)]

#[cfg(feature = "fem")]
pub(crate) mod assemble;
pub(crate) mod element_systems;
#[cfg(feature = "fem")]
pub(crate) mod solve;
