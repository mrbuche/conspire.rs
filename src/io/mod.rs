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

pub enum Encoding<P>
where
    P: AsRef<Path>,
{
    Ascii(P),
    Binary(P),
}

impl<P> AsRef<Path> for Encoding<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Encoding::Ascii(path) => path.as_ref(),
            Encoding::Binary(path) => path.as_ref(),
        }
    }
}

pub trait Write<P>
where
    P: AsRef<Path>,
{
    type Error;
    fn write(&self, path: P) -> Result<(), Self::Error>;
}
