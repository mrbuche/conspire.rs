#[cfg(test)]
mod test;

use super::unstructured::{WriteVtkUnstructured, data_array};
use crate::geometry::mesh::{Connectivity, Mesh};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result, Write},
    path::{Path, PathBuf},
};

pub trait WriteVtkMultiBlock<P>
where
    P: AsRef<Path>,
{
    fn write_vtk_multi_block(&self, output: P) -> Result<()>;
}

impl<const D: usize, P> WriteVtkMultiBlock<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_vtk_multi_block(&self, output: P) -> Result<()> {
        let path = output.as_ref();
        let dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "invalid VTK multiblock path"))?;
        let join = |name: &str| -> PathBuf {
            match dir {
                Some(dir) => dir.join(name),
                None => PathBuf::from(name),
            }
        };
        let volume_file = format!("{stem}.vtu");
        self.write_vtk_unstructured(join(&volume_file))?;
        let mut blocks = vec![("volume".to_string(), volume_file)];
        for (set, sides) in self.side_sets().iter().enumerate() {
            let label = self
                .side_set_numbers()
                .map_or_else(|| (set + 1).to_string(), |numbers| numbers[set].to_string());
            let side_set_file = format!("{stem}_side_set_{label}.vtp");
            write_side_set(self, sides, join(&side_set_file))?;
            blocks.push((format!("side_set_{label}"), side_set_file));
        }
        let mut file = BufWriter::new(File::create(path)?);
        writeln!(file, "<?xml version=\"1.0\"?>")?;
        writeln!(
            file,
            "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">"
        )?;
        writeln!(file, "  <vtkMultiBlockDataSet>")?;
        for (index, (name, file_name)) in blocks.iter().enumerate() {
            writeln!(
                file,
                "    <DataSet index=\"{index}\" name=\"{name}\" file=\"{file_name}\"/>"
            )?;
        }
        writeln!(file, "  </vtkMultiBlockDataSet>")?;
        writeln!(file, "</VTKFile>")?;
        Ok(())
    }
}

fn locate_element<const D: usize>(mesh: &Mesh<D>, element: usize) -> (&Connectivity, usize) {
    let mut offset = 0;
    for connectivity in mesh.iter() {
        let count = connectivity.number_of_elements();
        if element < offset + count {
            return (connectivity, element - offset);
        }
        offset += count;
    }
    panic!("side set references an out-of-range element {element}");
}

fn write_side_set<const D: usize>(
    mesh: &Mesh<D>,
    sides: &[(usize, usize)],
    output: impl AsRef<Path>,
) -> Result<()> {
    let coordinates = mesh.coordinates();
    let mut local_index = HashMap::new();
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut polys: Vec<Vec<usize>> = Vec::new();
    let mut lines: Vec<Vec<usize>> = Vec::new();
    for &(element, ordinal) in sides {
        let (connectivity, local) = locate_element(mesh, element);
        let nodes = connectivity
            .iter()
            .nth(local)
            .expect("side set references an out-of-range element");
        let face: Vec<usize> = connectivity.local_faces()[ordinal]
            .iter()
            .map(|&local_node| nodes[local_node])
            .map(|global| {
                *local_index.entry(global).or_insert_with(|| {
                    let index = points.len();
                    let mut point = [0.0; 3];
                    (0..D).for_each(|i| point[i] = coordinates[global][i]);
                    points.push(point);
                    index
                })
            })
            .collect();
        if face.len() == 2 {
            lines.push(face);
        } else {
            polys.push(face);
        }
    }
    let mut point_bytes = Vec::with_capacity(points.len() * 3 * 8);
    points.iter().for_each(|point| {
        point
            .iter()
            .for_each(|&value| point_bytes.extend_from_slice(&value.to_le_bytes()))
    });
    let (line_connectivity, line_offsets) = flatten(&lines);
    let (poly_connectivity, poly_offsets) = flatten(&polys);
    let mut file = BufWriter::new(File::create(output)?);
    writeln!(file, "<?xml version=\"1.0\"?>")?;
    writeln!(
        file,
        "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
    )?;
    writeln!(file, "  <PolyData>")?;
    writeln!(
        file,
        "    <Piece NumberOfPoints=\"{}\" NumberOfVerts=\"0\" NumberOfLines=\"{}\" NumberOfStrips=\"0\" NumberOfPolys=\"{}\">",
        points.len(),
        lines.len(),
        polys.len()
    )?;
    writeln!(file, "      <Points>")?;
    writeln!(
        file,
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">{}</DataArray>",
        data_array(&point_bytes)
    )?;
    writeln!(file, "      </Points>")?;
    if !lines.is_empty() {
        writeln!(file, "      <Lines>")?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\">{}</DataArray>",
            data_array(&line_connectivity)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">{}</DataArray>",
            data_array(&line_offsets)
        )?;
        writeln!(file, "      </Lines>")?;
    }
    if !polys.is_empty() {
        writeln!(file, "      <Polys>")?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"binary\">{}</DataArray>",
            data_array(&poly_connectivity)
        )?;
        writeln!(
            file,
            "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"binary\">{}</DataArray>",
            data_array(&poly_offsets)
        )?;
        writeln!(file, "      </Polys>")?;
    }
    writeln!(file, "    </Piece>")?;
    writeln!(file, "  </PolyData>")?;
    writeln!(file, "</VTKFile>")?;
    Ok(())
}

fn flatten(faces: &[Vec<usize>]) -> (Vec<u8>, Vec<u8>) {
    let mut connectivity = Vec::new();
    let mut offsets = Vec::new();
    let mut offset: i64 = 0;
    for face in faces {
        face.iter()
            .for_each(|&node| connectivity.extend_from_slice(&(node as i64).to_le_bytes()));
        offset += face.len() as i64;
        offsets.extend_from_slice(&offset.to_le_bytes());
    }
    (connectivity, offsets)
}
