//! Input/output library.

#[cfg(feature = "netcdf")]
mod netcdf;
pub mod npy;

use std::path::Path;

#[cfg(feature = "netcdf")]
pub use netcdf::{DefineVariable, GetVariable, NetCDF, PutVariable};

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
