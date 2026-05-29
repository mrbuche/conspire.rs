#[cfg(test)]
mod test;

use crate::geometry::{Write, mesh::MeshNew};
use std::{io::Error as ErrorIO, path::Path};

#[cfg(feature = "netcdf")]
use super::exodus::WriteExodus;

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

impl<const D: usize, P, T> Write<Output<P>> for MeshNew<D, T>
where
    P: AsRef<Path>,
    T: Copy + Into<i32>,
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
