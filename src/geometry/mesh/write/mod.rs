#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Write,
        mesh::{PrimitiveMesh, exodus::write as write_exodus},
    },
    math::Scalar,
};
use std::{io::Result as ResultIO, path::Path};

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Exodus(P),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Exodus(path) => path.as_ref(),
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
            Output::Exodus(path) => write_exodus(self, path),
        }
    }
}
