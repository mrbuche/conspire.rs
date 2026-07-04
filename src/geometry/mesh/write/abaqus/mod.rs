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

pub trait WriteAbaqus<P>
where
    P: AsRef<Path>,
{
    fn write_abaqus(&self, output: P) -> Result<()>;
}

fn abaqus_type(connectivity: &Connectivity) -> Result<&'static str> {
    Ok(match connectivity {
        Connectivity::Triangular(_) => "S3",
        Connectivity::Quadrilateral(_) => "S4",
        Connectivity::Tetrahedral(_) => "C3D4",
        Connectivity::Pyramidal(_) => "C3D5",
        Connectivity::Wedge(_) => "C3D6",
        Connectivity::Hexahedral(_) => "C3D8",
        Connectivity::Polygonal(_) | Connectivity::Polyhedral(_) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Abaqus writer does not support polygonal/polyhedral blocks",
            ));
        }
    })
}

impl<const D: usize, P> WriteAbaqus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_abaqus(&self, output: P) -> Result<()> {
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "*Heading")?;
        writeln!(file, " conspire mesh")?;
        writeln!(file, "*Node")?;
        let coordinates = self.coordinates();
        for node in 0..coordinates.len() {
            write!(file, "{}", node + 1)?;
            for i in 0..D {
                write!(file, ", {}", coordinates[node][i])?;
            }
            writeln!(file)?;
        }
        let mut element = 1;
        for (block, connectivity) in self.iter().enumerate() {
            let element_type = abaqus_type(connectivity)?;
            writeln!(
                file,
                "*Element, type={element_type}, elset=BLOCK{}",
                block + 1
            )?;
            for nodes in connectivity.iter() {
                write!(file, "{element}")?;
                for &node in nodes {
                    write!(file, ", {}", node + 1)?;
                }
                writeln!(file)?;
                element += 1;
            }
        }
        for (set, nodes) in self.node_sets().iter().enumerate() {
            writeln!(file, "*Nset, nset=NSET{}", set + 1)?;
            for chunk in nodes.chunks(16) {
                let line = chunk
                    .iter()
                    .map(|&node| (node + 1).to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(file, "{line}")?;
            }
        }
        Ok(())
    }
}
