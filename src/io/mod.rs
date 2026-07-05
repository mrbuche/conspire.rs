//! Input/output library.

#[cfg(feature = "netcdf")]
mod netcdf;
mod npy;
mod zip;

use std::path::Path;

#[cfg(feature = "netcdf")]
pub use netcdf::{DefineVariable, GetVariable, NetCDF, PutVariable};
pub use npy::{Npy, NpyType};
pub use zip::{Zip, ZipEntry};

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
