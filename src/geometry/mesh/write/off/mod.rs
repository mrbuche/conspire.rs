#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    math::Tensor,
};
use std::{
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

pub(crate) trait WriteOff<P>
where
    P: AsRef<Path>,
{
    fn write_off(&self, output: P) -> Result<()>;
}

impl<const D: usize, P> WriteOff<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_off(&self, output: P) -> Result<()> {
        if D != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                ".off holds 3D points but Mesh has D != 3",
            ));
        }
        if self.iter().any(|connectivity| {
            !matches!(
                connectivity,
                Connectivity::Triangular(_) | Connectivity::Quadrilateral(_)
            )
        }) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                ".off holds only triangular and quadrilateral faces",
            ));
        }
        let faces: usize = self
            .iter()
            .map(|connectivity| connectivity.number_of_elements())
            .sum();
        let coordinates = self.coordinates();
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "OFF")?;
        writeln!(file, "{} {faces} 0", coordinates.len())?;
        for node in 0..coordinates.len() {
            writeln!(
                file,
                "{} {} {}",
                coordinates[node][0], coordinates[node][1], coordinates[node][2]
            )?;
        }
        for connectivity in self.iter() {
            for element in connectivity.iter() {
                write!(file, "{}", element.len())?;
                for &node in element {
                    write!(file, " {node}")?;
                }
                writeln!(file)?;
            }
        }
        Ok(())
    }
}
