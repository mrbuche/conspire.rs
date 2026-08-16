#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        mesh::{
            Connectivity,
            tessellation::{D, Tessellation},
        },
    },
    io::Write,
    math::Tensor,
};
use std::{
    fs::File,
    io::{BufWriter, Error as ErrorIO, Write as WriteIO},
    path::Path,
};

pub enum Stl<P>
where
    P: AsRef<Path>,
{
    Ascii(P),
    Binary(P),
}

impl<P> AsRef<Path> for Stl<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Stl::Ascii(path) => path.as_ref(),
            Stl::Binary(path) => path.as_ref(),
        }
    }
}

impl<P> Write<Stl<P>> for Tessellation
where
    P: AsRef<Path>,
{
    type Error = ErrorIO;
    fn write(&self, output: Stl<P>) -> Result<(), Self::Error> {
        match output {
            Stl::Ascii(path) => self.write_stl_ascii(path)?,
            Stl::Binary(path) => self.write_stl_binary(path)?,
        }
        Ok(())
    }
}

impl Tessellation {
    fn for_each_facet<F>(&self, mut facet: F) -> Result<(), ErrorIO>
    where
        F: FnMut(&Direction<D>, [&Coordinate<D>; D]) -> Result<(), ErrorIO>,
    {
        self.mesh
            .connectivities()
            .iter()
            .zip(self.normals.iter())
            .try_for_each(|(connectivity, normals)| match connectivity {
                Connectivity::Triangular(triangles) => triangles
                    .iter()
                    .zip(normals.iter())
                    .try_for_each(|(nodes, normal)| {
                        facet(normal, nodes.map(|node| &self.mesh.coordinates()[node]))
                    }),
                _ => panic!("STL only supports triangular blocks"),
            })
    }
    fn write_stl_binary<P>(&self, path: P) -> Result<(), ErrorIO>
    where
        P: AsRef<Path>,
    {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&[0_u8; 80])?;
        writer.write_all(&(self.mesh.number_of_elements() as u32).to_le_bytes())?;
        self.for_each_facet(|normal, vertices| {
            normal.iter().try_for_each(|&component| {
                writer.write_all(&(component.value() as f32).to_le_bytes())
            })?;
            vertices.iter().try_for_each(|vertex| {
                vertex.iter().try_for_each(|&coordinate| {
                    writer.write_all(&(coordinate.value() as f32).to_le_bytes())
                })
            })?;
            writer.write_all(&0_u16.to_le_bytes())
        })?;
        writer.flush()
    }
    fn write_stl_ascii<P>(&self, path: P) -> Result<(), ErrorIO>
    where
        P: AsRef<Path>,
    {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(b"solid conspire\n")?;
        self.for_each_facet(|normal, vertices| {
            writeln!(
                writer,
                "  facet normal {} {} {}\n    outer loop",
                normal[0].value() as f32,
                normal[1].value() as f32,
                normal[2].value() as f32
            )?;
            vertices.iter().try_for_each(|vertex| {
                writeln!(
                    writer,
                    "      vertex {} {} {}",
                    vertex[0].value() as f32,
                    vertex[1].value() as f32,
                    vertex[2].value() as f32
                )
            })?;
            writer.write_all(b"    endloop\n  endfacet\n")
        })?;
        writer.write_all(b"endsolid conspire\n")?;
        writer.flush()
    }
}
