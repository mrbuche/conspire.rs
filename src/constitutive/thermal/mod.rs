//! Thermal constitutive models.

pub mod conduction;

use crate::mechanics::{HeatFlux, Scalar, TemperatureGradient};

use super::{Constitutive, Parameters};
use std::fmt::Debug;

/// Required methods for thermal constitutive models.
pub trait Thermal
where
    Self: Debug,
{
}
