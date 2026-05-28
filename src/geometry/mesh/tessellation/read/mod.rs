#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::tessellation::{D, Tessellation},
    },
    math::TensorVec,
};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Error as ErrorIO, Read},
    path::Path,
};

impl<T> TryFrom<&Path> for Tessellation<T>
where
    T: Copy + From<usize>,
{
    type Error = ErrorIO;
    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut header = [0u8; 80];
        reader.read_exact(&mut header)?;
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let triangle_count = u32::from_le_bytes(count_bytes) as usize;
        let mut connectivity = Vec::with_capacity(triangle_count);
        let mut normals = Coordinates::with_capacity(triangle_count);
        let mut vertex_map = HashMap::with_capacity(D * triangle_count);
        let mut unique_vertices_f32 = Vec::with_capacity(D * triangle_count);
        (0..triangle_count).try_for_each(|_| {
            let normal_f32 = read_vec3_f32(&mut reader)?;
            let v0 = read_vec3_f32(&mut reader)?;
            let v1 = read_vec3_f32(&mut reader)?;
            let v2 = read_vec3_f32(&mut reader)?;
            let mut attr = [0u8; 2];
            reader.read_exact(&mut attr)?;
            let _attribute_byte_count = u16::from_le_bytes(attr);
            let i0 = dedup_vertex(&mut vertex_map, &mut unique_vertices_f32, v0);
            let i1 = dedup_vertex(&mut vertex_map, &mut unique_vertices_f32, v1);
            let i2 = dedup_vertex(&mut vertex_map, &mut unique_vertices_f32, v2);
            connectivity.push([T::from(i0), T::from(i1), T::from(i2)]);
            normals.push(Coordinate::const_from([
                normal_f32[0] as f64,
                normal_f32[1] as f64,
                normal_f32[2] as f64,
            ]));
            Ok::<(), ErrorIO>(())
        })?;
        let coordinates: Coordinates<D> = unique_vertices_f32
            .into_iter()
            .map(|v| Coordinate::const_from([v[0] as f64, v[1] as f64, v[2] as f64]))
            .collect();
        let mesh = (connectivity, coordinates).into();
        Ok(Tessellation { mesh, normals })
    }
}

fn read_vec3_f32<R: Read>(reader: &mut R) -> Result<[f32; D], ErrorIO> {
    Ok([read_f32(reader)?, read_f32(reader)?, read_f32(reader)?])
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, ErrorIO> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn dedup_vertex(
    vertex_map: &mut HashMap<[u32; D], usize>,
    unique_vertices: &mut Vec<[f32; D]>,
    vertex: [f32; D],
) -> usize {
    let key = [
        vertex[0].to_bits(),
        vertex[1].to_bits(),
        vertex[2].to_bits(),
    ];
    if let Some(&index) = vertex_map.get(&key) {
        index
    } else {
        let index = unique_vertices.len();
        unique_vertices.push(vertex);
        vertex_map.insert(key, index);
        index
    }
}
