use crate::{
    geometry::{
        cad::{
            brep::test::{axis_aligned_box, unit_cube},
            sizing::FeatureSizing,
        },
        mesh::{Class, Connectivity, Fitting, Mesh, Verdict},
        ntree::Balancing,
    },
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

/// Distance from a point to the wireframe of the axis-aligned box `[0, extents]`.
fn distance_to_box_edges(point: [f64; 3], extents: [f64; 3]) -> f64 {
    let mut best = f64::INFINITY;
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        for &cu in &[0.0, extents[u]] {
            for &cv in &[0.0, extents[v]] {
                let along = (point[axis]).clamp(0.0, extents[axis]);
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
    // A rectangular box: the octree root is a cube of the longest side, so it
    // overhangs the geometry on the two shorter axes.
    let extents = [2.0, 4.0, 8.0];
    let brep = axis_aligned_box(extents);
    let sizing = FeatureSizing::of(&brep, 64, length(0.01), length(2.0), 0.25);
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
    let near_edges = |hex: &[usize; 8]| distance_to_box_edges(cell_center(hex, &mesh), extents);
    let smallest_cell = cells
        .iter()
        .min_by(|a, b| cell_size(a, &mesh).total_cmp(&cell_size(b, &mesh)))
        .unwrap();
    let largest_cell = cells
        .iter()
        .max_by(|a, b| cell_size(a, &mesh).total_cmp(&cell_size(b, &mesh)))
        .unwrap();
    assert!(near_edges(smallest_cell) < near_edges(largest_cell));

    // Padding zero, so the block fills the octree root cube: an 8x8x8 cube
    // centred on the box, overhanging the geometry on x and y.
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    let side = extents.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for axis in 0..3 {
        let center = 0.5 * extents[axis];
        assert!((low[axis] - (center - side / 2.0)).abs() < 1e-9);
        assert!((high[axis] - (center + side / 2.0)).abs() < 1e-9);
    }
    // The root cube genuinely hangs past the geometry on the short axes.
    assert!(low[0] < -1e-6 && high[0] > extents[0] + 1e-6);
    assert!(low[1] < -1e-6 && high[1] > extents[1] + 1e-6);
}

#[test]
fn max_levels_is_bounded() {
    let brep = unit_cube();
    let sizing = FeatureSizing::of(&brep, 16, length(0.01), length(2.0), 0.25);
    assert!(brep.sizing_octree(&sizing, 0, 0.0).is_err());
    assert!(brep.sizing_octree(&sizing, 16, 0.0).is_err());
}

#[test]
fn dual_background_classifies_the_dual_mesh() {
    let brep = unit_cube();
    let sizing = FeatureSizing::of(&brep, 16, length(0.01), length(2.0), 0.25);
    let (mesh, classes) = brep
        .dual_background(&sizing, 5, 0.1, Balancing::Strong(1))
        .unwrap();
    assert_eq!(hexes(&mesh).len(), classes.len());
    assert!(classes.contains(&Class::Inside));
    assert!(classes.contains(&Class::Cut));

    // The flood fill agrees with a direct test away from the boundary.
    let centroids = mesh.centroids();
    for (index, &class) in classes.iter().enumerate() {
        if class == Class::Cut {
            continue;
        }
        assert_eq!(
            class == Class::Inside,
            brep.encloses(&centroids[index]).unwrap(),
            "cell {index}"
        );
    }
}

#[test]
fn trim_hugs_the_geometry() {
    let extents = [2.0, 4.0, 8.0];
    let brep = axis_aligned_box(extents);
    let sizing = FeatureSizing::of(&brep, 64, length(0.05), length(1.0), 0.25);
    let (mesh, classes) = brep.trim(&sizing, 6, 0.1, Balancing::Strong(1)).unwrap();

    assert_eq!(hexes(&mesh).len(), classes.len());
    assert!(classes.iter().all(|&class| class != Class::Outside));
    assert!(classes.contains(&Class::Inside) && classes.contains(&Class::Cut));

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    // The padded root cube is [-3.4, 5.4] x [-2.4, 6.4] x [-0.4, 8.4]; the
    // trimmed block clings to the box, one boundary cell proud at most.
    assert!(low[0] > -1.0 && high[0] < 3.0);
    assert!(low[1] > -1.0 && high[1] < 5.0);
    for axis in 0..3 {
        assert!(low[axis] <= 1e-9 && high[axis] >= extents[axis] - 1e-9);
    }
}

#[test]
fn mesh_fits_the_graded_box() {
    let extents = [2.0, 4.0, 8.0];
    let brep = axis_aligned_box(extents);
    let sizing = FeatureSizing::of(&brep, 64, length(0.05), length(1.0), 0.25);
    let mesh = brep
        .mesh(&sizing, 6, 0.1, Balancing::Strong(1), Fitting::Soft)
        .unwrap();

    assert_eq!(mesh.connectivities().len(), 1);
    let jacobians = mesh.minimum_scaled_jacobians();
    assert!(
        jacobians[0].iter().all(|&j| j > 0.0),
        "inverted hex: worst scaled Jacobian {}",
        jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min)
    );

    // The graded, edge-refined dual fits onto the box faces to within a small
    // fraction of the coarsest boundary edge.
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    for axis in 0..3 {
        assert!(low[axis].abs() < 5e-3, "low[{axis}] = {}", low[axis]);
        assert!(
            (high[axis] - extents[axis]).abs() < 5e-3,
            "high[{axis}] = {}",
            high[axis]
        );
    }
}
