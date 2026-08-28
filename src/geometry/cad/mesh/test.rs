use crate::{
    geometry::{
        cad::{brep::test::unit_cube, sizing::FeatureSizing},
        mesh::{Connectivity, Mesh, Output, Vtk},
        ntree::Balancing,
    },
    io::{Write, write::Compression},
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn hexes(mesh: &Mesh<3>) -> Vec<[usize; 8]> {
    let Connectivity::Hexahedral(block) = &mesh.connectivities()[0] else {
        panic!("expected a hexahedral mesh");
    };
    block.iter().copied().collect()
}

fn cell_size(hex: &[usize; 8], mesh: &Mesh<3>) -> f64 {
    let coordinates = mesh.coordinates();
    (0..3)
        .map(|axis| {
            let values = hex.map(|node| coordinates[node][axis].value());
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().cloned().fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

fn cell_center(hex: &[usize; 8], mesh: &Mesh<3>) -> [f64; 3] {
    let coordinates = mesh.coordinates();
    let mut center = [0.0; 3];
    for &node in hex {
        for axis in 0..3 {
            center[axis] += coordinates[node][axis].value() / 8.0;
        }
    }
    center
}

/// Distance from a point to the wireframe of the unit cube.
fn distance_to_cube_edges(point: [f64; 3]) -> f64 {
    let mut best = f64::INFINITY;
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for &cu in &[0.0, 1.0] {
            for &cv in &[0.0, 1.0] {
                let along = (point[axis]).clamp(0.0, 1.0);
                let foot = {
                    let mut f = [0.0; 3];
                    f[axis] = along;
                    f[u] = cu;
                    f[v] = cv;
                    f
                };
                let d = ((point[0] - foot[0]).powi(2)
                    + (point[1] - foot[1]).powi(2)
                    + (point[2] - foot[2]).powi(2))
                .sqrt();
                best = best.min(d);
            }
        }
    }
    best
}

#[test]
fn sizing_field_grades_the_octree() {
    let brep = unit_cube();
    let sizing = FeatureSizing::of(&brep, 16, length(0.01), length(2.0), 0.25);
    let mesh = brep.sizing_octree(&sizing, 7, 0.0).unwrap();
    let cells = hexes(&mesh);

    assert!(
        cells.len() > 8,
        "octree barely refined: {} cells",
        cells.len()
    );
    assert!(cells.len() < 64 * 64 * 64);

    let sizes: Vec<f64> = cells.iter().map(|hex| cell_size(hex, &mesh)).collect();
    let smallest = sizes.iter().cloned().fold(f64::INFINITY, f64::min);
    let largest = sizes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(largest > smallest + 1e-9, "octree is uniform, not graded");

    // Small cells hug the edges; large cells sit in the interior.
    let near_edges = |hex: &[usize; 8]| distance_to_cube_edges(cell_center(hex, &mesh));
    let smallest_cell = cells
        .iter()
        .min_by(|a, b| cell_size(a, &mesh).total_cmp(&cell_size(b, &mesh)))
        .unwrap();
    let largest_cell = cells
        .iter()
        .max_by(|a, b| cell_size(a, &mesh).total_cmp(&cell_size(b, &mesh)))
        .unwrap();
    assert!(near_edges(smallest_cell) < near_edges(largest_cell));

    // Padding zero, so the block fills the cube exactly.
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    for axis in 0..3 {
        assert!(low[axis].abs() < 1e-9 && (high[axis] - 1.0).abs() < 1e-9);
    }

    mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
        "target/cad_cube_sizing_octree.vtu",
    ))))
    .unwrap();
}

#[test]
fn max_levels_is_bounded() {
    let brep = unit_cube();
    let sizing = FeatureSizing::of(&brep, 16, length(0.01), length(2.0), 0.25);
    assert!(brep.sizing_octree(&sizing, 0, 0.0).is_err());
    assert!(brep.sizing_octree(&sizing, 16, 0.0).is_err());
}

#[test]
fn tessellation_dual_background_still_runs() {
    let brep = unit_cube();
    let (mesh, classes) = brep.dual_background(Balancing::Strong(1), 4.0).unwrap();
    assert_eq!(hexes(&mesh).len(), classes.len());
    assert!(classes.contains(&crate::geometry::mesh::Class::Inside));
}
