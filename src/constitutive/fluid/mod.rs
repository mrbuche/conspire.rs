//! Fluid constitutive models.

pub mod viscous;

use super::Constitutive;
use crate::mechanics::Scalar;

/// Required methods for fluid constitutive models.
pub trait Fluid<P>
where
    Self: Constitutive<P>,
{
}
