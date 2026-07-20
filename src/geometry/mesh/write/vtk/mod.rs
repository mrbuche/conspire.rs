pub mod multi_block;
pub mod unstructured;

pub use unstructured::UnstructuredGrid;

use std::path::Path;

pub enum Vtk<P>
where
    P: AsRef<Path>,
{
    UnstructuredGrid(UnstructuredGrid<P>),
    MultiBlock(P),
}

impl<P> AsRef<Path> for Vtk<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Vtk::UnstructuredGrid(grid) => grid.as_ref(),
            Vtk::MultiBlock(path) => path.as_ref(),
        }
    }
}
