#[cfg(test)]
mod test;

#[cfg(feature = "netcdf")]
pub mod exodus;

use crate::{geometry::mesh::Mesh, io::Write};
use std::{io::Error as ErrorIO, path::Path};

#[cfg(feature = "netcdf")]
use self::exodus::WriteExodus;

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Abaqus(P),
    #[cfg(feature = "netcdf")]
    Exodus(P),
    Mesh(P),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Abaqus(path) => path.as_ref(),
            #[cfg(feature = "netcdf")]
            Output::Exodus(path) => path.as_ref(),
            Output::Mesh(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, P> Write<Output<P>> for Mesh<D>
where
    P: AsRef<Path>,
{
    type Error = ErrorIO;
    fn write(&self, output: Output<P>) -> Result<(), Self::Error> {
        match output {
            Output::Abaqus(_) => {}
            #[cfg(feature = "netcdf")]
            Output::Exodus(path) => self.write_exodus(path)?,
            Output::Mesh(_) => {}
        }
        Ok(())
    }
}
