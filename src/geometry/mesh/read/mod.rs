#[cfg(feature = "netcdf")]
pub mod exodus;
pub mod mesh;

#[cfg(feature = "netcdf")]
pub use self::exodus::ReadExodus;
pub use self::mesh::ReadMesh;

use crate::geometry::mesh::Mesh;
use std::{io::Error as ErrorIO, path::Path};

pub enum Input<P>
where
    P: AsRef<Path>,
{
    Abaqus(P),
    #[cfg(feature = "netcdf")]
    Exodus(P),
    Mesh(P),
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
            Input::Mesh(path) => path.as_ref(),
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
            Input::Abaqus(_) => {
                unimplemented!()
            }
            #[cfg(feature = "netcdf")]
            Input::Exodus(path) => Ok(Mesh::read_exodus(path)?),
            Input::Mesh(path) => Ok(Mesh::read_mesh(path)?),
        }
    }
}
