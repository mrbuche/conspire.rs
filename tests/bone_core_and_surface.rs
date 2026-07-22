#![cfg(feature = "geometry")]

use conspire::{
    geometry::{
        mesh::{Connectivity, Mesh, Output, Tessellation, Vtk},
        ntree::{Balancing, CurvatureSizing},
    },
    io::{Write, write::Compression},
    math::{Tensor, TensorVec},
};
use std::{array::from_fn, fs::File, io::Write as IoWrite, path::Path};

#[test]
fn bone_imprinted_network() {
    let Ok(tessellation) = Tessellation::try_from(Path::new("bone_tri.stl")) else {
        return;
    };
    let imprint = tessellation
        .imprinted_network(Balancing::Strong, 4.0, CurvatureSizing::default(), 8)
        .unwrap();
    let (core, surface) = (&imprint.core, &imprint.surface);
    let offset = core.number_of_nodes();
    let mut coordinates = core.coordinates().clone();
    let surface_coordinates = surface.coordinates();
    (0..surface_coordinates.len())
        .for_each(|node| coordinates.push(surface_coordinates[node].clone()));
    let hexes: Vec<[usize; 8]> = core
        .iter()
        .flatten()
        .map(|hex| from_fn(|i| hex[i]))
        .collect();
    let triangles: Vec<[usize; 3]> = surface
        .iter()
        .flatten()
        .map(|triangle| from_fn(|i| triangle[i] + offset))
        .collect();
    let mesh = Mesh::from((
        vec![
            Connectivity::Hexahedral(hexes.into()),
            Connectivity::Triangular(triangles.into()),
        ],
        coordinates,
    ));
    mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
        "bone_network.vtu",
    ))))
    .unwrap();
    let number_of_triangles = surface.number_of_elements();
    let mut labels = vec![0usize; number_of_triangles];
    imprint
        .patches
        .iter()
        .enumerate()
        .for_each(|(face, patch)| patch.iter().for_each(|&triangle| labels[triangle] = face));
    let surface_triangles: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let mut file = File::create("bone_imprint.vtk").unwrap();
    writeln!(file, "# vtk DataFile Version 3.0").unwrap();
    writeln!(file, "imprinted surface patches").unwrap();
    writeln!(file, "ASCII").unwrap();
    writeln!(file, "DATASET UNSTRUCTURED_GRID").unwrap();
    writeln!(file, "POINTS {} double", surface_coordinates.len()).unwrap();
    (0..surface_coordinates.len()).for_each(|node| {
        let point = &surface_coordinates[node];
        writeln!(file, "{} {} {}", point[0], point[1], point[2]).unwrap();
    });
    writeln!(
        file,
        "CELLS {} {}",
        number_of_triangles,
        4 * number_of_triangles
    )
    .unwrap();
    surface_triangles.iter().for_each(|triangle| {
        writeln!(file, "3 {} {} {}", triangle[0], triangle[1], triangle[2]).unwrap();
    });
    writeln!(file, "CELL_TYPES {number_of_triangles}").unwrap();
    (0..number_of_triangles).for_each(|_| writeln!(file, "5").unwrap());
    writeln!(file, "CELL_DATA {number_of_triangles}").unwrap();
    writeln!(file, "SCALARS face int 1").unwrap();
    writeln!(file, "LOOKUP_TABLE default").unwrap();
    labels
        .iter()
        .for_each(|label| writeln!(file, "{label}").unwrap());
    let mut points = Vec::new();
    let mut lines: Vec<Vec<usize>> = Vec::new();
    imprint.paths.iter().for_each(|path| {
        let start = points.len();
        path.iter()
            .for_each(|&vertex| points.push(surface_coordinates[vertex].clone()));
        lines.push((start..points.len()).collect());
    });
    let mut file = File::create("bone_imprint_paths.vtk").unwrap();
    writeln!(file, "# vtk DataFile Version 3.0").unwrap();
    writeln!(file, "imprinted network paths").unwrap();
    writeln!(file, "ASCII").unwrap();
    writeln!(file, "DATASET POLYDATA").unwrap();
    writeln!(file, "POINTS {} double", points.len()).unwrap();
    points.iter().for_each(|point| {
        writeln!(file, "{} {} {}", point[0], point[1], point[2]).unwrap();
    });
    let size: usize = lines.iter().map(|line| line.len() + 1).sum();
    writeln!(file, "LINES {} {}", lines.len(), size).unwrap();
    lines.iter().for_each(|line| {
        write!(file, "{}", line.len()).unwrap();
        line.iter()
            .for_each(|index| write!(file, " {index}").unwrap());
        writeln!(file).unwrap();
    });
    println!(
        "core hexes: {}, surface triangles: {}, faces: {}, paths: {}",
        core.number_of_elements(),
        surface.number_of_elements(),
        imprint.faces.len(),
        imprint.paths.len()
    );
}
