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

pub trait WriteMedit<P>
where
    P: AsRef<Path>,
{
    fn write_medit(&self, output: P) -> Result<()>;
}

// A Medit section: keyword plus its (element nodes, 1-based block) records.
type Section<'a> = (&'static str, Vec<(&'a [usize], usize)>);

fn section(connectivity: &Connectivity) -> Result<&'static str> {
    match connectivity {
        Connectivity::Triangular(_) => Ok("Triangles"),
        Connectivity::Quadrilateral(_) => Ok("Quadrilaterals"),
        Connectivity::Tetrahedral(_) => Ok("Tetrahedra"),
        Connectivity::Pyramidal(_) => Ok("Pyramids"),
        Connectivity::Wedge(_) => Ok("Prisms"),
        Connectivity::Hexahedral(_) => Ok("Hexahedra"),
        Connectivity::Polygonal(_) | Connectivity::Polyhedral(_) => Err(Error::new(
            ErrorKind::Unsupported,
            "Medit .mesh does not support polygonal/polyhedral blocks",
        )),
    }
}

impl<const D: usize, P> WriteMedit<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_medit(&self, output: P) -> Result<()> {
        if D != 2 && D != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Medit .mesh supports only 2D or 3D meshes",
            ));
        }
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "MeshVersionFormatted 2")?;
        writeln!(file, "Dimension {D}")?;
        let coordinates = self.coordinates();
        writeln!(file, "Vertices")?;
        writeln!(file, "{}", coordinates.len())?;
        for node in 0..coordinates.len() {
            for i in 0..D {
                write!(file, "{} ", coordinates[node][i])?;
            }
            writeln!(file, "0")?;
        }
        let mut sections: Vec<Section<'_>> = Vec::new();
        for (block, connectivity) in self.iter().enumerate() {
            let keyword = section(connectivity)?;
            let index = match sections.iter().position(|(found, _)| *found == keyword) {
                Some(index) => index,
                None => {
                    sections.push((keyword, Vec::new()));
                    sections.len() - 1
                }
            };
            for element in connectivity.iter() {
                sections[index].1.push((element, block + 1));
            }
        }
        for (keyword, elements) in &sections {
            writeln!(file, "{keyword}")?;
            writeln!(file, "{}", elements.len())?;
            for (nodes, reference) in elements {
                for &node in *nodes {
                    write!(file, "{} ", node + 1)?;
                }
                writeln!(file, "{reference}")?;
            }
        }
        writeln!(file, "End")?;
        Ok(())
    }
}
