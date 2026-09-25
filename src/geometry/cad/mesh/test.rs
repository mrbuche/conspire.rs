use crate::{
    geometry::{
        Coordinate,
        cad::{
            brep::test::{
                axis_aligned_box, ball, capped_cylinder, cone, notched_bore_block,
                notched_bore_pocket, torus, unit_cube,
            },
            sizing::FeatureSizing,
        },
        mesh::{Class, Connectivity, Fitting, Mesh, Output, Verdict, Vtk},
        ntree::Balancing,
        solid::{Solid, SolidOracle},
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
    let sizing = FeatureSizing::of(&brep, 64, Some(length(0.01)), Some(length(2.0)), Some(0.25));
    let mesh = brep.sizing_octree(&sizing, Some(7), 0.0).unwrap();
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
    let sizing = FeatureSizing::of(&brep, 16, Some(length(0.01)), Some(length(2.0)), Some(0.25));
    assert!(brep.sizing_octree(&sizing, Some(0), 0.0).is_err());
    assert!(brep.sizing_octree(&sizing, Some(16), 0.0).is_err());
    // `None` is uncapped: the coarse `maximum` still settles it well short of
    // the tree's own depth limit.
    assert!(brep.sizing_octree(&sizing, None, 0.0).is_ok());
}

#[test]
fn dual_background_classifies_the_dual_mesh() {
    let brep = unit_cube();
    let sizing = FeatureSizing::of(&brep, 16, Some(length(0.01)), Some(length(2.0)), Some(0.25));
    let (mesh, classes) = brep
        .dual_background(&sizing, Some(5), 0.1, Balancing::Strong(1))
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
    let sizing = FeatureSizing::of(&brep, 64, Some(length(0.05)), Some(length(1.0)), Some(0.25));
    let (mesh, classes) = brep
        .trim(&sizing, Some(6), 0.1, Balancing::Strong(1))
        .unwrap();

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
fn meshes_a_capped_cylinder_through_the_analytic_oracle() {
    let brep = capped_cylinder(2.0, 5.0);
    let sizing = FeatureSizing::of(&brep, 32, Some(length(0.1)), Some(length(1.0)), Some(0.25));
    let mesh = brep
        .mesh(&sizing, Some(6), 0.1, Balancing::Strong(1), Fitting::Soft)
        .unwrap();

    assert_eq!(mesh.connectivities().len(), 1);
    let jacobians = mesh.minimum_scaled_jacobians();
    assert!(
        jacobians[0].iter().all(|&j| j > 0.0),
        "inverted hex: worst scaled Jacobian {}",
        jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min)
    );

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    // Clings to the r = 2, z in [0, 5] cylinder, a boundary cell proud at most.
    for axis in 0..2 {
        assert!(low[axis] > -2.5 && low[axis] < -1.0);
        assert!(high[axis] < 2.5 && high[axis] > 1.0);
    }
    assert!(low[2].abs() < 0.6 && (high[2] - 5.0).abs() < 0.6);
}

#[test]
fn crease_curves_pull_rim_nodes_onto_the_exact_circle() {
    // A boundary node near the cap/wall rim, fit by nearest-face alone, is
    // free to land anywhere on the cap's tangent plane -- its radius is
    // unconstrained there, only z is pinned. With the rim's crease curve
    // wired in explicitly via Solid::mesh_with_creases (Solid::mesh itself
    // no longer applies a crease constraint by default -- see its doc), a
    // node this close to the rim should land on the exact circle: both z and
    // radius pinned together, not just z.
    let brep = capped_cylinder(2.0, 5.0);
    let sizing = FeatureSizing::of(&brep, 32, Some(length(0.1)), Some(length(0.4)), Some(0.25));
    let creases = brep.creases();
    let mesh = brep
        .mesh_with_creases(
            &sizing,
            Some(6),
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
            &creases,
        )
        .unwrap();

    let mut checked = 0;
    for coordinate in mesh.coordinates() {
        let radius = (coordinate[0].value().powi(2) + coordinate[1].value().powi(2)).sqrt();
        let z = coordinate[2].value();
        let near_a_rim = z.abs() < 0.15 || (z - 5.0).abs() < 0.15;
        let plausibly_on_the_rim = (radius - 2.0).abs() < 0.3;
        if near_a_rim && plausibly_on_the_rim {
            checked += 1;
            // Without the crease constraint (measured directly: forcing
            // Brep::creases to return nothing on this same mesh) the worst
            // deviations here are 0.019 / 0.016 -- nearest-face alone gets
            // rim nodes only that close. With it, 0.0069 / 0.0071.
            assert!((radius - 2.0).abs() < 0.012, "radius {radius} at z {z}");
            assert!(
                z.abs() < 0.012 || (z - 5.0).abs() < 0.012,
                "z {z} at radius {radius}"
            );
        }
    }
    assert!(checked > 0, "no rim-region node found to check");
}

#[test]
fn mesh_fits_the_graded_box() {
    let extents = [2.0, 4.0, 8.0];
    let brep = axis_aligned_box(extents);
    let sizing = FeatureSizing::of(&brep, 64, Some(length(0.05)), Some(length(1.0)), Some(0.25));
    let mesh = brep
        .mesh(&sizing, Some(6), 0.1, Balancing::Strong(1), Fitting::Soft)
        .unwrap();

    assert_eq!(mesh.connectivities().len(), 1);
    let jacobians = mesh.minimum_scaled_jacobians();
    assert!(
        jacobians[0].iter().all(|&j| j > 0.0),
        "inverted hex: worst scaled Jacobian {}",
        jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min)
    );

    // The graded, edge-refined dual fits onto the box faces to within a small
    // fraction of the coarsest boundary edge. `mesh()` applies no crease
    // constraint (see `crease_curves_pull_rim_nodes_onto_the_exact_circle`
    // for the constrained variant, via `mesh_with_creases`).
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for axis in 0..3 {
            low[axis] = low[axis].min(coordinate[axis].value());
            high[axis] = high[axis].max(coordinate[axis].value());
        }
    }
    for axis in 0..3 {
        assert!(low[axis].abs() < 1e-2, "low[{axis}] = {}", low[axis]);
        assert!(
            (high[axis] - extents[axis]).abs() < 1e-2,
            "high[{axis}] = {}",
            high[axis]
        );
    }
}

/// Dumps meshed curved B-reps to `target/*.vtu` for eyeballing in ParaView.
/// Ignored: nothing reads them back.
#[test]
#[ignore = "writes target/cad_*.vtu for manual inspection"]
fn dump_curved_brep_meshes() {
    let cases: [(&str, Mesh<3>); 4] = [
        (
            "target/cad_capped_cylinder.vtu",
            capped_cylinder(2.0, 5.0)
                .mesh(
                    &FeatureSizing::of(
                        &capped_cylinder(2.0, 5.0),
                        48,
                        Some(length(0.15)),
                        Some(length(1.0)),
                        Some(0.2),
                    ),
                    Some(7),
                    0.1,
                    Balancing::Strong(1),
                    Fitting::Soft,
                )
                .unwrap(),
        ),
        (
            "target/cad_cone.vtu",
            cone(3.0, 1.0, 5.0)
                .mesh(
                    &FeatureSizing::of(
                        &cone(3.0, 1.0, 5.0),
                        48,
                        Some(length(0.15)),
                        Some(length(1.0)),
                        Some(0.2),
                    ),
                    Some(7),
                    0.1,
                    Balancing::Strong(1),
                    Fitting::Soft,
                )
                .unwrap(),
        ),
        (
            // No feature edges (the seam is an artifact): a uniform field, so
            // `maximum` sets the resolution.
            "target/cad_sphere.vtu",
            ball(3.0)
                .mesh(
                    &FeatureSizing::of(
                        &ball(3.0),
                        48,
                        Some(length(0.15)),
                        Some(length(0.5)),
                        Some(0.2),
                    ),
                    Some(7),
                    0.1,
                    Balancing::Strong(1),
                    Fitting::Soft,
                )
                .unwrap(),
        ),
        (
            "target/cad_torus.vtu",
            torus(4.0, 1.5)
                .mesh(
                    &FeatureSizing::of(
                        &torus(4.0, 1.5),
                        48,
                        Some(length(0.12)),
                        Some(length(0.4)),
                        Some(0.2),
                    ),
                    Some(7),
                    0.1,
                    Balancing::Strong(1),
                    Fitting::Soft,
                )
                .unwrap(),
        ),
    ];
    for (path, mesh) in &cases {
        let worst = mesh.minimum_scaled_jacobians()[0]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "{path}: {} hexes, worst scaled Jacobian {worst:.4}",
            hexes(mesh).len()
        );
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(path))))
            .unwrap();
    }
}

/// Face-connected component count, the size of the smallest component, and
/// the nodes on boundary faces (a quad used by exactly one hex).
fn topology(mesh: &Mesh<3>) -> (usize, usize, Vec<usize>) {
    use std::collections::HashMap;
    const FACES: [[usize; 4]; 6] = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let hexes = hexes(mesh);
    let mut parent: Vec<usize> = (0..hexes.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut seen: HashMap<[usize; 4], (usize, usize)> = HashMap::new();
    for (index, hex) in hexes.iter().enumerate() {
        for face in FACES {
            let mut key = face.map(|corner| hex[corner]);
            key.sort_unstable();
            match seen.get_mut(&key) {
                Some((first, count)) => {
                    *count += 1;
                    let (a, b) = (find(&mut parent, *first), find(&mut parent, index));
                    parent[a] = b;
                }
                None => {
                    seen.insert(key, (index, 1));
                }
            }
        }
    }
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for index in 0..hexes.len() {
        *sizes.entry(find(&mut parent, index)).or_default() += 1;
    }
    let mut boundary: Vec<usize> = seen
        .into_iter()
        .filter(|(_, (_, count))| *count == 1)
        .flat_map(|(key, _)| key)
        .collect();
    boundary.sort_unstable();
    boundary.dedup();
    (
        sizes.len(),
        sizes.values().copied().min().unwrap_or(0),
        boundary,
    )
}

/// Nodes lying strictly inside a hex they do not belong to: a positive-Jacobian
/// element pushed through a neighbour, which per-element quality cannot see.
fn penetrating_nodes(mesh: &Mesh<3>) -> usize {
    use std::collections::HashMap;
    const TETS: [[usize; 4]; 6] = [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    let hexes = hexes(mesh);
    let position = |node: usize| -> [f64; 3] {
        std::array::from_fn(|axis| mesh.coordinates()[node][axis].value())
    };
    let cell = hexes
        .iter()
        .map(|hex| cell_size(hex, mesh))
        .fold(0.0_f64, f64::max);
    let key = |p: [f64; 3]| -> [i64; 3] { p.map(|x| (x / cell).floor() as i64) };
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (index, hex) in hexes.iter().enumerate() {
        let points = hex.map(position);
        let low = key(std::array::from_fn(|k| {
            points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min)
        }));
        let high = key(std::array::from_fn(|k| {
            points
                .iter()
                .map(|p| p[k])
                .fold(f64::NEG_INFINITY, f64::max)
        }));
        for i in low[0]..=high[0] {
            for j in low[1]..=high[1] {
                for k in low[2]..=high[2] {
                    grid.entry([i, j, k]).or_default().push(index);
                }
            }
        }
    }
    let determinant = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> f64 {
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    let sub = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| a[k] - b[k]) };
    let inside = |p: [f64; 3], hex: &[usize; 8]| -> bool {
        let epsilon = 1e-6 * cell;
        TETS.iter().any(|tet| {
            let v = tet.map(|corner| position(hex[corner]));
            let volume = determinant(sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]));
            if volume.abs() < 1e-14 {
                return false;
            }
            let weights = [
                determinant(sub(v[1], p), sub(v[2], p), sub(v[3], p)),
                determinant(sub(p, v[0]), sub(v[2], v[0]), sub(v[3], v[0])),
                determinant(sub(v[1], v[0]), sub(p, v[0]), sub(v[3], v[0])),
                determinant(sub(v[1], v[0]), sub(v[2], v[0]), sub(p, v[0])),
            ];
            weights
                .iter()
                .all(|w| w / volume > epsilon / cell.max(1e-12) && w / volume < 1.0)
        })
    };
    let mut used: Vec<usize> = hexes.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let mut count = 0;
    for node in used {
        let p = position(node);
        let hit = grid.get(&key(p)).is_some_and(|candidates| {
            candidates
                .iter()
                .any(|&index| !hexes[index].contains(&node) && inside(p, &hexes[index]))
        });
        if hit {
            count += 1;
        }
    }
    count
}

/// `(hex count, low corner, high corner)` of every face-connected component.
fn component_boxes(mesh: &Mesh<3>) -> Vec<(usize, [f64; 3], [f64; 3])> {
    use std::collections::HashMap;
    let hexes = hexes(mesh);
    let mut owner: HashMap<[usize; 4], usize> = HashMap::new();
    let mut parent: Vec<usize> = (0..hexes.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (index, hex) in hexes.iter().enumerate() {
        for face in [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ] {
            let mut key = face.map(|corner| hex[corner]);
            key.sort_unstable();
            if let Some(&other) = owner.get(&key) {
                let (a, b) = (find(&mut parent, other), find(&mut parent, index));
                parent[a] = b;
            } else {
                owner.insert(key, index);
            }
        }
    }
    let mut boxes: HashMap<usize, (usize, [f64; 3], [f64; 3])> = HashMap::new();
    for (index, hex) in hexes.iter().enumerate() {
        let root = find(&mut parent, index);
        let entry = boxes
            .entry(root)
            .or_insert((0, [f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
        entry.0 += 1;
        for &node in hex {
            for axis in 0..3 {
                let value = mesh.coordinates()[node][axis].value();
                entry.1[axis] = entry.1[axis].min(value);
                entry.2[axis] = entry.2[axis].max(value);
            }
        }
    }
    let mut boxes: Vec<_> = boxes.into_values().collect();
    boxes.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    boxes
}

#[test]
fn notched_bore_block_is_a_valid_solid() {
    let (gap, radius) = (0.3, 1.6);
    let brep = notched_bore_block(gap, radius, 2.0, std::f64::consts::FRAC_PI_4);
    let oracle = brep.oracle().unwrap();
    let sd = |x: f64, y: f64| oracle.signed_distance(&Coordinate::const_from([x, y, 1.0]));
    let apex = 3.4 - radius - gap;
    assert!(sd(1.0, 5.0) > 0.0, "solid interior");
    assert!(sd(3.0, 3.4) < 0.0, "inside the bore");
    assert!(sd(3.0, 0.3) < 0.0, "inside the notch");
    assert!(sd(3.0, apex + gap / 2.0) > 0.0, "inside the ligament");
    assert!((sd(3.0, apex + gap / 2.0) - gap / 2.0).abs() < 1e-6);
    assert!(sd(-1.0, 3.0) < 0.0, "outside the block");
}

#[test]
fn notched_bore_pocket_is_a_valid_solid() {
    let (gap, radius) = (0.3, 1.6);
    let brep = notched_bore_pocket(gap, radius, 2.0, std::f64::consts::FRAC_PI_4, [0.5, 1.5]);
    let oracle = brep.oracle().unwrap();
    let sd = |x: f64, y: f64, z: f64| oracle.signed_distance(&Coordinate::const_from([x, y, z]));
    let apex = 3.4 - radius - gap;
    assert!(sd(1.0, 5.0, 1.0) > 0.0, "solid interior");
    assert!(sd(3.0, 3.4, 1.0) < 0.0, "inside the bore");
    assert!(sd(3.0, 0.3, 1.0) < 0.0, "inside the pocket");
    assert!(sd(3.0, 0.3, 0.25) > 0.0, "solid below the pocket");
    assert!(sd(3.0, 0.3, 1.75) > 0.0, "solid above the pocket");
    assert!((sd(3.0, apex + gap / 2.0, 1.0) - gap / 2.0).abs() < 1e-6);
    assert!(sd(-1.0, 3.0, 1.0) < 0.0, "outside the block");
    assert!(!brep.crease_curves().is_empty());
}

/// Meshes `notched_bore_block` over ligament thicknesses `gap / h` with and
/// without the crease constraint and reports validity; writes
/// `target/notched_g{ratio}_{plain,crease}.vtu`.
#[test]
#[ignore = "diagnostic: reports and writes target/notched_*.vtu"]
fn probe_notched_bore_block() {
    let (radius, height) = (1.6, 1.2);
    let modes: &[bool] = if std::env::var("PROBE_GRADED").is_ok() {
        &[false, true]
    } else {
        &[false]
    };
    for (h, ratio, graded) in [0.3, 0.55, 0.8].into_iter().flat_map(|h| {
        [1.0, 0.5, 0.25]
            .into_iter()
            .flat_map(move |ratio| modes.iter().map(move |&graded| (h, ratio, graded)))
    }) {
        let gap = ratio * h;
        let brep = notched_bore_pocket(
            gap,
            radius,
            height,
            std::f64::consts::FRAC_PI_4,
            [0.3 * height, 0.7 * height],
        );
        let minimum = (!graded).then(|| length(h));
        let mut sizing = FeatureSizing::of(&brep, 24, minimum, Some(length(h)), Some(0.2))
            .with_proximity(&brep, 3)
            .unwrap();
        if graded {
            sizing = sizing.with_feature_separation(&brep, 3).unwrap();
        }
        let oracle = brep.oracle().unwrap();
        let ligament = [3.0, 3.4 - radius - gap / 2.0, 0.6 * height];
        for with_crease in [false, true] {
            let creases = if with_crease { brep.creases() } else { vec![] };
            let mesh = brep
                .mesh_with_creases(
                    &sizing,
                    None,
                    0.1,
                    Balancing::Strong(1),
                    Fitting::Soft,
                    &creases,
                )
                .unwrap();
            let jacobians = mesh.minimum_scaled_jacobians();
            let worst = jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min);
            let inverted = jacobians[0].iter().filter(|&&j| j <= 0.0).count();
            let outside = hexes(&mesh)
                .iter()
                .filter(|hex| {
                    let [x, y, z] = cell_center(hex, &mesh);
                    oracle.signed_distance(&Coordinate::const_from([x, y, z])) < -0.05 * h
                })
                .count();
            let tag = if with_crease { "crease" } else { "plain" };
            let mode = if graded { "graded" } else { "flat" };
            let across = hexes(&mesh)
                .iter()
                .filter_map(|hex| {
                    let coordinates = mesh.coordinates();
                    let bounds = |axis: usize| {
                        let values = hex.map(|node| coordinates[node][axis].value());
                        (
                            values.iter().cloned().fold(f64::INFINITY, f64::min),
                            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        )
                    };
                    (0..3)
                        .all(|axis| {
                            let (low, high) = bounds(axis);
                            low <= ligament[axis] && ligament[axis] <= high
                        })
                        .then(|| gap / (bounds(1).1 - bounds(1).0))
                })
                .fold(0.0_f64, f64::max);
            let (components, smallest, boundary) = topology(&mesh);
            let surface_error = boundary
                .iter()
                .map(|&node| {
                    let c = &mesh.coordinates()[node];
                    oracle
                        .signed_distance(&Coordinate::const_from([
                            c[0].value(),
                            c[1].value(),
                            c[2].value(),
                        ]))
                        .abs()
                })
                .fold(0.0_f64, f64::max);
            eprintln!(
                "h {h} g/h {ratio} {mode:>6} {tag:>6}: {} hexes, worst SJ {worst:.4}, {inverted} inverted, \
                 {outside} outside, {components} comp (smallest {smallest}), \
                 boundary err {:.3} h, ligament cells across {across:.2}, \
                 {} penetrating nodes, volume {:.4} of exact",
                hexes(&mesh).len(),
                surface_error / h,
                penetrating_nodes(&mesh),
                mesh.volumes()[0].iter().sum::<f64>()
                    / (36.0 * height
                        - std::f64::consts::PI * radius * radius * height
                        - (3.4 - radius - gap).powi(2) * 0.4 * height)
            );
            if components > 1 && with_crease {
                for (size, low, high) in component_boxes(&mesh) {
                    eprintln!(
                        "    component of {size} hexes: x {:.2}..{:.2}, y {:.2}..{:.2}, z {:.2}..{:.2}",
                        low[0], high[0], low[1], high[1], low[2], high[2]
                    );
                }
            }
            mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
                &format!("target/notched_h{h}_g{ratio}_{mode}_{tag}.vtu"),
            ))))
            .unwrap();
        }
    }
}
