pub mod abaqus;
#[cfg(feature = "netcdf")]
pub mod exodus;
pub mod medit;
pub mod vtk;

pub use self::abaqus::ReadAbaqus;
#[cfg(feature = "netcdf")]
pub use self::exodus::ReadExodus;
pub use self::medit::ReadMedit;
pub use self::vtk::{multi_block::ReadVtkMultiBlock, unstructured::ReadVtkUnstructured};

use crate::geometry::mesh::Mesh;
use std::{io::Error as ErrorIO, path::Path};

pub enum Input<P>
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

impl<P> AsRef<Path> for Input<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Input::Abaqus(path) => path.as_ref(),
            #[cfg(feature = "netcdf")]
            Input::Exodus(path) => path.as_ref(),
            Input::Medit(path) => path.as_ref(),
            Input::VtkUnstructured(path) => path.as_ref(),
            Input::VtkMultiBlock(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, P> TryFrom<Input<P>> for Mesh<D>
where
    P: AsRef<Path>,
{
    type Error = ErrorIO;
    fn try_from(input: Input<P>) -> Result<Self, Self::Error> {
        match input {
            Input::Abaqus(path) => Ok(Mesh::read_abaqus(path)?),
            #[cfg(feature = "netcdf")]
            Input::Exodus(path) => Ok(Mesh::read_exodus(path)?),
            Input::Medit(path) => Ok(Mesh::read_medit(path)?),
            Input::VtkUnstructured(path) => Ok(Mesh::read_vtk_unstructured(path)?),
            Input::VtkMultiBlock(path) => Ok(Mesh::read_vtk_multi_block(path)?),
        }
    }
}
