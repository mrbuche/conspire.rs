#[cfg(test)]
pub mod test;

use crate::geometry::{Write, mesh::PrimitiveMesh};
use std::{io::Result as ResultIO, path::Path};

#[cfg(feature = "netcdf")]
use crate::geometry::mesh::exodus::write as write_exodus;

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

impl<const D: usize, const I: usize, const M: usize, const N: usize, P, T> Write<Output<P>>
    for PrimitiveMesh<D, I, M, N, T>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    fn write(&self, output: Output<P>) -> ResultIO<()> {
        match output {
            Output::Abaqus(_) => todo!(),
            #[cfg(feature = "netcdf")]
            Output::Exodus(path) => write_exodus(self, path),
            Output::Mesh(_) => todo!(),
        }
    }
}
