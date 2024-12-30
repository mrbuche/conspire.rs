//! Thermal constitutive models.

pub mod conduction;

use crate::mechanics::{HeatFlux, Scalar, TemperatureGradient};

use super::{Constitutive, Parameters};

/// Required methods for thermal constitutive models.
pub trait Thermal<'a>
where
    Self: Constitutive<'a>,
{
}
