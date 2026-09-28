#[cfg(feature = "fem")]
pub(crate) mod assemble;
pub(crate) mod element;
pub(crate) mod solid;
#[cfg(feature = "fem")]
pub(crate) mod solve;
