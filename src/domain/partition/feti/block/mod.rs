#[cfg(feature = "fem")]
pub(crate) mod assemble;
pub(crate) mod element;
#[cfg(feature = "fem")]
pub(crate) mod solve;
