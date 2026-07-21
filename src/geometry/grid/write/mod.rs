mod spn;
mod vti;

use crate::{
    geometry::grid::Grid,
    io::{Npy, NpyType, Write, write::Compression},
};
use std::{fmt::Display, io::Error as ErrorIO, path::Path};

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Npy(P),
    Spn(P),
    Vti(Compression<P>),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Npy(path) => path.as_ref(),
            Output::Spn(path) => path.as_ref(),
            Output::Vti(vti) => vti.as_ref(),
        }
    }
}

impl<const D: usize, T, P> Write<Output<P>> for Grid<D, T>
where
    P: AsRef<Path>,
    T: NpyType + Display,
{
    type Error = ErrorIO;
    fn write(&self, output: Output<P>) -> Result<(), Self::Error> {
        match output {
            Output::Npy(path) => Npy {
                data: self.data().to_vec(),
                shape: self.nel().to_vec(),
                fortran_order: self.is_col_major(),
            }
            .write(path)?,
            Output::Spn(path) => spn::write(self, path)?,
            Output::Vti(Compression::On(path)) => vti::write(self, path, true)?,
            Output::Vti(Compression::Off(path)) => vti::write(self, path, false)?,
        }
        Ok(())
    }
}
