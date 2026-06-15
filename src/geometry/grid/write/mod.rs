mod vti;

use crate::{
    geometry::grid::Grid,
    io::{Npy, NpyType, Write},
};
use std::{io::Error as ErrorIO, path::Path};

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Npy(P),
    Vti(P),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Npy(path) => path.as_ref(),
            Output::Vti(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, T, P> Write<Output<P>> for Grid<D, T>
where
    P: AsRef<Path>,
    T: NpyType,
{
    type Error = ErrorIO;
    fn write(&self, output: Output<P>) -> Result<(), Self::Error> {
        match output {
            Output::Npy(path) => Npy {
                data: self.data().to_vec(),
                shape: self.nel().to_vec(),
                fortran_order: true,
            }
            .write(path)?,
            Output::Vti(path) => vti::write(self, path)?,
        }
        Ok(())
    }
}
