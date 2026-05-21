#[cfg(test)]
pub mod test;

use crate::{
    geometry::{Write, mesh::tessellation::Tessellation},
    math::Tensor,
};
use std::{
    fs::File,
    io::{BufWriter, Result as ResultIO, Write as WriteIO},
    path::Path,
};

impl<const I: usize, T, P> Write<P> for Tessellation<I, T>
where
    P: AsRef<Path>,
    T: Copy + Into<usize>,
{
    fn write(&self, path: P) -> ResultIO<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&[0_u8; 80])?;
        writer.write_all(&(self.mesh.connectivity.len() as u32).to_le_bytes())?;
        self.mesh
            .connectivity
            .iter()
            .zip(self.normals.iter())
            .try_for_each(|(nodes, normal)| {
                normal.iter().try_for_each(|&component| {
                    writer.write_all(&(component as f32).to_le_bytes())
                })?;
                nodes.iter().try_for_each(|&node| {
                    self.mesh.coordinates[node.into()]
                        .iter()
                        .try_for_each(|&coordinate| {
                            writer.write_all(&(coordinate as f32).to_le_bytes())
                        })
                })?;
                writer.write_all(&0_u16.to_le_bytes())
            })?;
        writer.flush()
    }
}
