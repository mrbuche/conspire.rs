//! Whether a matrix of a model, such as its mass or its heat capacity, is lumped or consistent.

/// The lumped matrix of the model, which is diagonal and needs a division to be inverted.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lumped;

/// The consistent matrix of the model, which is sparse and needs a solve to be inverted.
#[derive(Clone, Copy, Debug, Default)]
pub struct Consistent;
