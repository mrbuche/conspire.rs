#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh},
    io::write::{data_array, data_array_compressed},
    math::Tensor,
};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::Path,
};

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
