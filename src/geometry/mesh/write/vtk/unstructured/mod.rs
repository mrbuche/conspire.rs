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

pub trait WriteVtkUnstructured<P>
where
    P: AsRef<Path>,
{
    fn write_vtk_unstructured(&self, output: P) -> Result<()>;
}

fn cell_type(connectivity: &Connectivity) -> Result<u8> {
    Ok(match connectivity {
        Connectivity::Triangular(_) => 5,
        Connectivity::Quadrilateral(_) => 9,
        Connectivity::Tetrahedral(_) => 10,
        Connectivity::Hexahedral(_) => 12,
        Connectivity::Wedge(_) => 13,
        Connectivity::Pyramidal(_) => 14,
        Connectivity::Polygonal(_) | Connectivity::Polyhedral(_) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "VTU writer does not support polygonal/polyhedral blocks",
            ));
        }
    })
}

impl<const D: usize, P> WriteVtkUnstructured<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_vtk_unstructured(&self, output: P) -> Result<()> {
        if D != 2 && D != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "VTU supports only 2D or 3D meshes",
            ));
        }
        let coordinates = self.coordinates();
        let mut points = Vec::with_capacity(coordinates.len() * 3 * 8);
        for node in 0..coordinates.len() {
            for i in 0..3 {
                let value = if i < D { coordinates[node][i] } else { 0.0 };
                points.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();
        let mut types = Vec::new();
        let mut offset: i64 = 0;
        for block in self.iter() {
            let cell = cell_type(block)?;
            for element in block.iter() {
                for &node in element {
                    connectivity.extend_from_slice(&(node as i64).to_le_bytes());
                }
                offset += element.len() as i64;
                offsets.extend_from_slice(&offset.to_le_bytes());
                types.push(cell);
            }
        }
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "<?xml version=\"1.0\"?>")?;
        writeln!(
            file,
            "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
        )?;
        writeln!(file, "  <UnstructuredGrid>")?;
        writeln!(
            file,
            "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
            coordinates.len(),
            types.len()
        )?;
        if !self.node_sets().is_empty() {
            writeln!(file, "      <PointData>")?;
            for (set, nodes) in self.node_sets().iter().enumerate() {
                let mut flags = vec![0_u8; coordinates.len()];
                for &node in nodes {
                    flags[node] = 1;
                }
                writeln!(
                    file,
                    "        <DataArray type=\"UInt8\" Name=\"NodeSet{}\" format=\"binary\">{}</DataArray>",
                    set + 1,
                    data_array(&flags)
                )?;
            }
            writeln!(file, "      </PointData>")?;
        }
        writeln!(file, "      <Points>")?;
        writeln!(
            file,
            "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">{}</DataArray>",
            data_array(&points)
        )?;
        writeln!(file, "      </Points>")?;
        writeln!(file, "      <Cells>")?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\">{}</DataArray>",
            data_array(&connectivity)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">{}</DataArray>",
            data_array(&offsets)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">{}</DataArray>",
            data_array(&types)
        )?;
        writeln!(file, "      </Cells>")?;
        writeln!(file, "    </Piece>")?;
        writeln!(file, "  </UnstructuredGrid>")?;
        writeln!(file, "</VTKFile>")?;
        Ok(())
    }
}

pub(super) fn data_array(data: &[u8]) -> String {
    let mut buffer = Vec::with_capacity(8 + data.len());
    buffer.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buffer.extend_from_slice(data);
    base64(&buffer)
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let triple = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(ALPHABET[(triple >> 18 & 63) as usize] as char);
        out.push(ALPHABET[(triple >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
