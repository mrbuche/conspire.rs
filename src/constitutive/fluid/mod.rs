//! Fluid constitutive models.

pub mod hyperviscous;
pub mod plastic;
pub mod viscoplastic;
pub mod viscous;

/// Required methods for fluid constitutive models.
pub trait Fluid {}
