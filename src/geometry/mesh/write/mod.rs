#[cfg(test)]
mod test;

pub mod abaqus;
#[cfg(feature = "netcdf")]
pub mod exodus;
pub mod medit;
pub mod vtk;

use crate::{geometry::mesh::Mesh, io::Write};
use std::{io::Error as ErrorIO, path::Path};

use self::abaqus::WriteAbaqus;
#[cfg(feature = "netcdf")]
use self::exodus::WriteExodus;
use self::medit::WriteMedit;
use self::vtk::{multi_block::WriteVtkMultiBlock, unstructured::WriteVtkUnstructured};

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Abaqus(P),
    #[cfg(feature = "netcdf")]
    Exodus(P),
    Medit(P),
    VtkUnstructured(P),
    VtkMultiBlock(P),
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
            Output::Medit(path) => path.as_ref(),
            Output::VtkUnstructured(path) => path.as_ref(),
            Output::VtkMultiBlock(path) => path.as_ref(),
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
            Output::Abaqus(path) => self.write_abaqus(path)?,
            #[cfg(feature = "netcdf")]
            Output::Exodus(path) => self.write_exodus(path)?,
            Output::Medit(path) => self.write_medit(path)?,
            Output::VtkUnstructured(path) => self.write_vtk_unstructured(path)?,
            Output::VtkMultiBlock(path) => self.write_vtk_multi_block(path)?,
        }
        Ok(())
    }
}
