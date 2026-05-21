#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates, Write, mesh::tessellation::Tessellation},
    math::{Tensor, TensorVec},
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Error as ErrorIO, Read, Result as ResultIO, Write as WriteIO},
    path::Path,
};

impl<const I: usize, T> TryFrom<&Path> for Tessellation<I, T>
where
    T: From<usize>,
{
    type Error = ErrorIO;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut header = [0u8; 80];
        reader.read_exact(&mut header)?;
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let triangle_count = u32::from_le_bytes(count_bytes) as usize;

        // need to efficiently merge nodes

        let mut coordinates =
            Coordinates::<3, I>::from(Vec::<Coordinate<_, _>>::with_capacity(3 * triangle_count));
        let mut connectivity = Vec::with_capacity(triangle_count);
        let mut normals =
            Coordinates::<3, I>::from(Vec::<Coordinate<_, _>>::with_capacity(triangle_count));

        for tri in 0..triangle_count {
            let normal = read_vec3::<I, _>(&mut reader)?;
            let v0 = read_vec3::<I, _>(&mut reader)?;
            let v1 = read_vec3::<I, _>(&mut reader)?;
            let v2 = read_vec3::<I, _>(&mut reader)?;

            let mut attr = [0u8; 2];
            reader.read_exact(&mut attr)?;
            let _attribute_count = u16::from_le_bytes(attr);

            let base = 3 * tri;
            coordinates.push(v0);
            coordinates.push(v1);
            coordinates.push(v2);

            connectivity.push([T::from(base), T::from(base + 1), T::from(base + 2)]);

            normals.push(normal);
        }

        let mesh = (connectivity, coordinates).into();
        Ok(Tessellation { mesh, normals })
    }
}

fn read_vec3<const I: usize, R: Read>(reader: &mut R) -> Result<Coordinate<3, I>, ErrorIO> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    let x = f32::from_le_bytes(bytes) as f64;
    reader.read_exact(&mut bytes)?;
    let y = f32::from_le_bytes(bytes) as f64;
    reader.read_exact(&mut bytes)?;
    let z = f32::from_le_bytes(bytes) as f64;
    Ok(Coordinate::const_from([x, y, z]))
}

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
