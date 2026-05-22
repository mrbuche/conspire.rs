#[cfg(test)]
mod test;

use crate::geometry::{Write, mesh::Mesh};
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
            Output::Exodus(p) => p.as_ref(),
        }
    }
}

impl<const D: usize, const I: usize, const M: usize, P, T> Write<Output<P>> for Mesh<D, I, M, T>
where
    P: AsRef<Path>,
{
    fn write(&self, output: Output<P>) -> ResultIO<()> {
        let [xs, ys, zs] = self.coordinates.into();
        todo!()
    }
}
