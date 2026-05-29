#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Write,
        mesh::{Connectivity, tessellation::Tessellation},
    },
    math::Tensor,
};
use std::{
    fs::File,
    io::{BufWriter, Error as ErrorIO, Write as WriteIO},
    path::Path,
};

impl<P> Write<P> for Tessellation
where
    P: AsRef<Path>,
{
    type Error = ErrorIO;
    fn write(&self, path: P) -> Result<(), Self::Error> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&[0_u8; 80])?;
        writer.write_all(&(self.mesh.number_of_elements() as u32).to_le_bytes())?;
        self.mesh
            .connectivities
            .iter()
            .zip(self.normals.iter())
            .try_for_each(|(connectivity, normals)| match connectivity {
                Connectivity::Triangular(triangles) => triangles
                    .iter()
                    .zip(normals.iter())
                    .try_for_each(|(nodes, normal)| {
                        normal.iter().try_for_each(|&component| {
                            writer.write_all(&(component as f32).to_le_bytes())
                        })?;
                        nodes.iter().try_for_each(|&node| {
                            self.mesh.coordinates[node]
                                .iter()
                                .try_for_each(|&coordinate| {
                                    writer.write_all(&(coordinate as f32).to_le_bytes())
                                })
                        })?;
                        writer.write_all(&0_u16.to_le_bytes())
                    }),
                _ => panic!("STL only supports triangular blocks"),
            })?;
        writer.flush()
    }
}
