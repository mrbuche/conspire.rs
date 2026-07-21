pub mod multi_block;
pub mod unstructured;

use crate::io::write::Compression;
use std::path::Path;

pub enum Vtk<P>
where
    P: AsRef<Path>,
{
    UnstructuredGrid(Compression<P>),
    MultiBlock(Compression<P>),
}

impl<P> AsRef<Path> for Vtk<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Vtk::UnstructuredGrid(grid) => grid.as_ref(),
            Vtk::MultiBlock(multi_block) => multi_block.as_ref(),
        }
    }
}
