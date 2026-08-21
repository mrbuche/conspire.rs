use crate::geometry::ntree::node::cell::Cell;
use crate::geometry::ntree::node::slot::Slot;
pub(super) mod htg;

use crate::{
    geometry::ntree::Orthotree,
    io::{Write, write::Compression},
};
use std::{io::Error as ErrorIO, path::Path};

use htg::WriteHtg;

pub enum Output<P>
where
    P: AsRef<Path>,
{
    Htg(Compression<P>),
}

impl<P> AsRef<Path> for Output<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Output::Htg(htg) => htg.as_ref(),
        }
    }
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, P> Write<Output<P>>
    for Orthotree<D, L, M, N, T, U>
where
    P: AsRef<Path>,
    T: Cell,
    U: Slot,
{
    type Error = ErrorIO;
    fn write(&self, output: Output<P>) -> Result<(), Self::Error> {
        match output {
            Output::Htg(Compression::On(path)) => self.write_htg_compressed(path)?,
            Output::Htg(Compression::Off(path)) => self.write_htg(path)?,
        }
        Ok(())
    }
}
