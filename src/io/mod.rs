//! Input/output library.

mod deflate;
#[cfg(feature = "netcdf")]
mod netcdf;
mod npy;
mod vtk;
mod zip;

use std::path::Path;

pub use deflate::{adler32, deflate, inflate, zlib_decode, zlib_encode};
#[cfg(feature = "netcdf")]
pub use netcdf::{DefineVariable, GetVariable, NetCDF, PutVariable};
pub use npy::{Npy, NpyType};
pub use vtk::{invalid, read, unsupported, write};
pub use zip::{Zip, ZipEntry};

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
