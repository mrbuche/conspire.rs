#[cfg(test)]
mod test;

pub mod abaqus;
#[cfg(feature = "netcdf")]
pub mod exodus;
pub mod medit;
pub mod vtk;

use crate::{
    geometry::mesh::Mesh,
    io::{Write, write::Compression},
};
use std::{io::Error as ErrorIO, path::Path};

use self::abaqus::WriteAbaqus;
#[cfg(feature = "netcdf")]
use self::exodus::WriteExodus;
use self::medit::WriteMedit;
use self::vtk::{Vtk, multi_block::WriteVtkMultiBlock, unstructured::WriteVtkUnstructured};

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Abaqus(P),
    #[cfg(feature = "netcdf")]
    Exodus(P),
    Medit(P),
    Vtk(Vtk<P>),
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
            Output::Vtk(vtk) => vtk.as_ref(),
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
            Output::Vtk(Vtk::UnstructuredGrid(Compression::On(path))) => {
                self.write_vtk_unstructured_compressed(path)?
            }
            Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))) => {
                self.write_vtk_unstructured(path)?
            }
            Output::Vtk(Vtk::MultiBlock(Compression::On(path))) => {
                self.write_vtk_multi_block_compressed(path)?
            }
            Output::Vtk(Vtk::MultiBlock(Compression::Off(path))) => {
                self.write_vtk_multi_block(path)?
            }
        }
        Ok(())
    }
}
