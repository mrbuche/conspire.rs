#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    io::deflate::zlib_encode,
    math::Tensor,
};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

const COMPRESSION_BLOCK_SIZE: usize = 32768;

pub trait WriteVtkUnstructured<P>
where
    P: AsRef<Path>,
{
    fn write_vtk_unstructured(&self, output: P) -> Result<()>;
    fn write_vtk_unstructured_compressed(&self, output: P) -> Result<()>;
}

pub enum UnstructuredGrid<P>
where
    P: AsRef<Path>,
{
    Compressed(P),
    Uncompressed(P),
}

impl<P> AsRef<Path> for UnstructuredGrid<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            UnstructuredGrid::Compressed(path) => path.as_ref(),
            UnstructuredGrid::Uncompressed(path) => path.as_ref(),
        }
    }
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
        self.write_vtk_unstructured_impl(output, false)
    }
    fn write_vtk_unstructured_compressed(&self, output: P) -> Result<()> {
        self.write_vtk_unstructured_impl(output, true)
    }
}

impl<const D: usize> Mesh<D> {
    fn write_vtk_unstructured_impl<P: AsRef<Path>>(&self, output: P, compress: bool) -> Result<()> {
        if D != 2 && D != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "VTU supports only 2D or 3D meshes",
            ));
        }
        let array = |data: &[u8]| -> String {
            if compress {
                data_array_compressed(data)
            } else {
                data_array(data)
            }
        };
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
        let mut faces = Vec::new();
        let mut faceoffsets = Vec::new();
        let mut offset: i64 = 0;
        let mut face_offset: i64 = 0;
        let mut has_polyhedra = false;
        for block in self.iter() {
            if let Connectivity::Polyhedral(poly) = block {
                has_polyhedra = true;
                let faces_nodes = poly.faces_nodes();
                for element_faces in poly.iter() {
                    let mut seen = HashSet::new();
                    let mut unique = Vec::new();
                    for &face in element_faces {
                        for &node in &faces_nodes[face] {
                            if seen.insert(node) {
                                unique.push(node);
                            }
                        }
                    }
                    unique.iter().for_each(|&node| {
                        connectivity.extend_from_slice(&(node as i64).to_le_bytes())
                    });
                    offset += unique.len() as i64;
                    offsets.extend_from_slice(&offset.to_le_bytes());
                    types.push(42_u8);
                    faces.extend_from_slice(&(element_faces.len() as i64).to_le_bytes());
                    face_offset += 1;
                    for &face in element_faces {
                        let nodes = &faces_nodes[face];
                        faces.extend_from_slice(&(nodes.len() as i64).to_le_bytes());
                        face_offset += 1;
                        nodes.iter().for_each(|&node| {
                            faces.extend_from_slice(&(node as i64).to_le_bytes());
                            face_offset += 1;
                        });
                    }
                    faceoffsets.extend_from_slice(&face_offset.to_le_bytes());
                }
            } else {
                let cell = cell_type(block)?;
                for element in block.iter() {
                    for &node in element {
                        connectivity.extend_from_slice(&(node as i64).to_le_bytes());
                    }
                    offset += element.len() as i64;
                    offsets.extend_from_slice(&offset.to_le_bytes());
                    types.push(cell);
                    faces.extend_from_slice(&0_i64.to_le_bytes());
                    face_offset += 1;
                    faceoffsets.extend_from_slice(&face_offset.to_le_bytes());
                }
            }
        }
        let mut file = BufWriter::new(File::create(output)?);
        writeln!(file, "<?xml version=\"1.0\"?>")?;
        if compress {
            writeln!(
                file,
                "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">"
            )?;
        } else {
            writeln!(
                file,
                "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
            )?;
        }
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
                    array(&flags)
                )?;
            }
            writeln!(file, "      </PointData>")?;
        }
        writeln!(file, "      <Points>")?;
        writeln!(
            file,
            "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">{}</DataArray>",
            array(&points)
        )?;
        writeln!(file, "      </Points>")?;
        writeln!(file, "      <Cells>")?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\">{}</DataArray>",
            array(&connectivity)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">{}</DataArray>",
            array(&offsets)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">{}</DataArray>",
            array(&types)
        )?;
        if has_polyhedra {
            writeln!(
                file,
                "        <DataArray type=\"Int64\" Name=\"faces\" format=\"binary\">{}</DataArray>",
                array(&faces)
            )?;
            writeln!(
                file,
                "        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"binary\">{}</DataArray>",
                array(&faceoffsets)
            )?;
        }
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

fn data_array_compressed(data: &[u8]) -> String {
    let compressed_blocks: Vec<Vec<u8>> = data
        .chunks(COMPRESSION_BLOCK_SIZE)
        .map(zlib_encode)
        .collect();
    let num_blocks = compressed_blocks.len() as u64;
    let last_block_size =
        data.len() as u64 - num_blocks.saturating_sub(1) * COMPRESSION_BLOCK_SIZE as u64;
    let mut buffer = Vec::with_capacity(24 + compressed_blocks.iter().map(Vec::len).sum::<usize>());
    buffer.extend_from_slice(&num_blocks.to_le_bytes());
    buffer.extend_from_slice(&(COMPRESSION_BLOCK_SIZE as u64).to_le_bytes());
    buffer.extend_from_slice(&last_block_size.to_le_bytes());
    for block in &compressed_blocks {
        buffer.extend_from_slice(&(block.len() as u64).to_le_bytes());
    }
    for block in &compressed_blocks {
        buffer.extend_from_slice(block);
    }
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
