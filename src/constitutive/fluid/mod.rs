//! Fluid constitutive models.

pub mod viscous;

use crate::mechanics::Scalar;

/// Required methods for fluid constitutive models.
pub trait Fluid {}
