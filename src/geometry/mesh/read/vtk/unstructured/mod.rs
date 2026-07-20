#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    io::{
        invalid,
        read::{attribute, data_array, encoding, floats, integers, region, tag},
        unsupported,
    },
};
use std::{
    fs::read_to_string,
    io::{ErrorKind, Result},
    path::Path,
};

pub trait ReadVtkUnstructured<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_vtk_unstructured(input: P) -> Result<Self>;
}

impl<const D: usize, P> ReadVtkUnstructured<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_vtk_unstructured(input: P) -> Result<Self> {
        if D != 2 && D != 3 {
            return Err(unsupported("VTU supports only 2D or 3D meshes"));
        }
        let text = match read_to_string(input) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::InvalidData => {
                return Err(unsupported(
                    "raw appended VTU is not supported (use ascii or binary)",
                ));
            }
            Err(error) => return Err(error),
        };
        let header = tag(&text, "<VTKFile")?;
        if attribute(header, "type") != Some("UnstructuredGrid") {
            return Err(invalid("VTU is not an UnstructuredGrid".into()));
        }
        if matches!(attribute(header, "byte_order"), Some(order) if order != "LittleEndian") {
            return Err(unsupported("big-endian VTU is not supported"));
        }
        let encoding = encoding(header)?;

        let points_region = region(&text, "Points")?;
        let points = floats(&data_array(points_region, None)?, &encoding)?;
        let components = attribute(tag(points_region, "<DataArray")?, "NumberOfComponents")
            .and_then(|n| n.parse().ok())
            .unwrap_or(3);

        let cells_region = region(&text, "Cells")?;
        let connectivity = integers(&data_array(cells_region, Some("connectivity"))?, &encoding)?;
        let offsets = integers(&data_array(cells_region, Some("offsets"))?, &encoding)?;
        let types = integers(&data_array(cells_region, Some("types"))?, &encoding)?;
        let cell_faces = if cells_region.contains("Name=\"faces\"") {
            let faces = integers(&data_array(cells_region, Some("faces"))?, &encoding)?;
            let faceoffsets = integers(&data_array(cells_region, Some("faceoffsets"))?, &encoding)?;
            decode_faces(&faces, &faceoffsets)?
        } else {
            vec![Vec::new(); types.len()]
        };

        let coordinates: Coordinates<D> = points
            .chunks(components)
            .map(|point| std::array::from_fn(|i| point[i]).into())
            .collect();
        let mut mesh = Mesh::<D>::from((
            blocks(&connectivity, &offsets, &types, &cell_faces)?,
            coordinates,
        ));
        if let Ok(point_data) = region(&text, "PointData") {
            let mut node_sets = Vec::new();
            let mut set = 1;
            while point_data.contains(&format!("Name=\"NodeSet{set}\"")) {
                let flags = integers(
                    &data_array(point_data, Some(&format!("NodeSet{set}")))?,
                    &encoding,
                )?;
                node_sets.push(
                    flags
                        .iter()
                        .enumerate()
                        .filter_map(|(node, &flag)| (flag != 0).then_some(node))
                        .collect(),
                );
                set += 1;
            }
            if !node_sets.is_empty() {
                mesh.set_node_sets(node_sets.into());
            }
        }
        Ok(mesh)
    }
}

fn blocks(
    connectivity: &[i64],
    offsets: &[i64],
    types: &[i64],
    faces: &[Vec<Vec<usize>>],
) -> Result<Vec<Connectivity>> {
    let mut cells: Vec<(i64, &[i64])> = Vec::with_capacity(types.len());
    let mut start = 0;
    for (cell, &end) in offsets.iter().enumerate() {
        cells.push((types[cell], &connectivity[start..end as usize]));
        start = end as usize;
    }
    let mut blocks = Vec::new();
    let mut from = 0;
    while from < cells.len() {
        let mut to = from;
        while to < cells.len() && cells[to].0 == cells[from].0 {
            to += 1;
        }
        blocks.push(block(cells[from].0, &cells[from..to], &faces[from..to])?);
        from = to;
    }
    Ok(blocks)
}

fn block(
    cell_type: i64,
    cells: &[(i64, &[i64])],
    faces: &[Vec<Vec<usize>>],
) -> Result<Connectivity> {
    Ok(match cell_type {
        5 => Connectivity::Triangular(arrays::<3>(cells)?.into()),
        9 => Connectivity::Quadrilateral(arrays::<4>(cells)?.into()),
        10 => Connectivity::Tetrahedral(arrays::<4>(cells)?.into()),
        12 => Connectivity::Hexahedral(arrays::<8>(cells)?.into()),
        13 => Connectivity::Wedge(arrays::<6>(cells)?.into()),
        14 => Connectivity::Pyramidal(arrays::<5>(cells)?.into()),
        42 => polyhedral(faces),
        other => return Err(invalid(format!("unsupported VTK cell type: {other}"))),
    })
}

fn polyhedral(faces: &[Vec<Vec<usize>>]) -> Connectivity {
    let mut elements_faces = Vec::with_capacity(faces.len());
    let mut faces_nodes = Vec::new();
    for element_faces in faces {
        elements_faces.push(
            element_faces
                .iter()
                .map(|face| {
                    let index = faces_nodes.len();
                    faces_nodes.push(face.clone());
                    index
                })
                .collect(),
        );
    }
    Connectivity::Polyhedral((elements_faces, faces_nodes).into())
}

fn decode_faces(faces: &[i64], faceoffsets: &[i64]) -> Result<Vec<Vec<Vec<usize>>>> {
    let mut cells = Vec::with_capacity(faceoffsets.len());
    let mut start = 0_usize;
    for &end in faceoffsets {
        let end = end as usize;
        let mut index = start;
        let num_faces = faces[index] as usize;
        index += 1;
        let mut cell_faces = Vec::with_capacity(num_faces);
        for _ in 0..num_faces {
            let num_points = faces[index] as usize;
            index += 1;
            cell_faces.push(
                faces[index..index + num_points]
                    .iter()
                    .map(|&p| p as usize)
                    .collect(),
            );
            index += num_points;
        }
        if index != end {
            return Err(invalid("faces/faceoffsets are inconsistent".into()));
        }
        cells.push(cell_faces);
        start = end;
    }
    Ok(cells)
}

fn arrays<const N: usize>(cells: &[(i64, &[i64])]) -> Result<Vec<[usize; N]>> {
    cells
        .iter()
        .map(|(_, nodes)| {
            if nodes.len() != N {
                return Err(invalid(format!(
                    "cell has {} nodes, expected {N}",
                    nodes.len()
                )));
            }
            Ok(std::array::from_fn(|i| nodes[i] as usize))
        })
        .collect()
}
