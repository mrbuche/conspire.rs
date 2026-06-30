pub mod htg;

use crate::{geometry::ntree::Orthotree, io::Write, math::Scalar};
use std::{io::Error as ErrorIO, path::Path};

use self::htg::WriteHtg;

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Htg(P),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Htg(path) => path.as_ref(),
        }
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, P> Write<Output<P>>
    for Orthotree<D, L, M, N, T, U>
where
    P: AsRef<Path>,
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    type Error = ErrorIO;
    fn write(&self, output: Output<P>) -> Result<(), Self::Error> {
        match output {
            Output::Htg(path) => self.write_htg(path)?,
        }
        Ok(())
    }
}
