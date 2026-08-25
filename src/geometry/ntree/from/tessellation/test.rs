use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Fitting, Mesh, Output, Tessellation, Vtk, test::sphere},
        ntree::node::{Node, cell::Cell},
        ntree::{
            Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing, SeparationSizing,
            Sizing,
        },
    },
    io::{Write, write::Compression},
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::{array::from_fn, collections::HashMap, path::Path};

fn curvature(tolerance: Quantity<Length>) -> CurvatureSizing {
    CurvatureSizing {
        tolerance: Some(tolerance),
        ..Default::default()
    }
}

#[test]
fn tighter_curvature_tolerance_refines_more() {
    let tessellation = sphere(12, 24, 2.0);
    let scale = 4.0;
    let loose = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0)),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    let medium = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-2)),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    let tight = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-3)),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    assert!(medium.len() > loose.len());
    assert!(tight.len() > medium.len());
}

#[test]
fn default_curvature_sizing_disables_curvature_refinement() {
    let tessellation = sphere(4, 8, 2.0);
    let scale = 4.0;
    let without = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-3)),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    let with_default = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    assert!(with_default.len() <= without.len());
}

fn tessellate(points: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Tessellation {
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

fn cube(divisions: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut lookup = HashMap::<[usize; 3], usize>::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut node = |grid: [usize; 3]| {
        *lookup.entry(grid).or_insert_with(|| {
            points.push(from_fn::<_, 3, _>(|axis| {
                grid[axis] as f64 / divisions as f64
            }));
            points.len() - 1
        })
    };
    for axis in 0..3 {
        let (along, across) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in [0, divisions] {
            for i in 0..divisions {
                for j in 0..divisions {
                    let corner = |one: usize, two: usize| {
                        let mut grid = [0; 3];
                        grid[axis] = side;
                        grid[along] = i + one;
                        grid[across] = j + two;
                        grid
                    };
                    let (a, b) = (node(corner(0, 0)), node(corner(1, 0)));
                    let (c, d) = (node(corner(1, 1)), node(corner(0, 1)));
                    if side == 0 {
                        faces.extend([[a, c, b], [a, d, c]])
                    } else {
                        faces.extend([[a, b, c], [a, c, d]])
                    }
                }
            }
        }
    }
    (points, faces)
}

fn pored_cube(radius: f64) -> Tessellation {
    let (mut points, mut faces) = cube(1);
    let pore = sphere(12, 24, radius);
    let offset = points.len();
    pore.mesh()
        .coordinates()
        .iter()
        .for_each(|point| points.push(from_fn::<_, 3, _>(|axis| point[axis].value() + 0.5)));
    pore.mesh()
        .connectivities()
        .iter()
        .flatten()
        .for_each(|element| {
            faces.push([
                element[0] + offset,
                element[2] + offset,
                element[1] + offset,
            ])
        });
    tessellate(points, faces)
}

fn refinement(tessellation: &Tessellation, curvature: CurvatureSizing) -> (f64, Vec<[f64; 3]>) {
    let mut octree = Octree::<u16, usize>::from_features(
        tessellation,
        5.0,
        curvature,
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let cells: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            element.iter().for_each(|&node| {
                let point = &background.coordinates()[node];
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis].value());
                    max[axis] = max[axis].max(point[axis].value());
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            (
                size,
                from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis])),
            )
        })
        .collect();
    let finest = cells
        .iter()
        .map(|&(size, _)| size)
        .fold(f64::INFINITY, f64::min);
    (
        finest,
        cells
            .iter()
            .filter(|&&(size, _)| size < finest * 1.5)
            .map(|&(_, center)| center)
            .collect(),
    )
}

#[test]
fn a_pore_refines_without_dragging_the_cube_down() {
    let radius = 0.1;
    let (finest, cells) = refinement(&pored_cube(radius), curvature(Quantity::new(1.0e-3)));
    assert!(finest < radius, "the pore is resolved at all");
    cells.iter().for_each(|center| {
        let distance = (0..3)
            .map(|axis| (center[axis] - 0.5).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            distance < 3.0 * radius,
            "a finest cell sits at {center:?}, away from the pore"
        )
    });
}

#[test]
fn a_tolerance_does_not_refine_a_polyhedron() {
    let (points, faces) = cube(4);
    let cube = tessellate(points, faces);
    let (without, _) = refinement(&cube, CurvatureSizing::default());
    let (tight, _) = refinement(&cube, curvature(Quantity::new(1.0e-4)));
    assert_eq!(
        tight, without,
        "no tolerance is tight enough to curve a flat face"
    );
}

#[test]
fn refuses_a_depth_no_cell_length_can_index() {
    let tessellation = sphere(4, 8, 2.0);
    assert_eq!(
        Octree::<u16, usize>::from_features(
            &tessellation,
            1.0e6,
            CurvatureSizing::default(),
            SeparationSizing::default(),
            0
        )
        .err(),
        Some("sizing field exceeds maximum octree depth")
    );
    assert!(
        Octree::<u16, usize>::from_features(
            &tessellation,
            4.0,
            CurvatureSizing::default(),
            SeparationSizing::default(),
            0
        )
        .is_ok()
    );
}

#[test]
fn a_wider_cell_carries_the_whole_pipeline() {
    let tessellation = sphere(4, 8, 2.0);
    let narrow = Octree::<u16, usize>::from_features(
        &tessellation,
        4.0,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    let mut wide = Octree::<u32, usize>::from_features(
        &tessellation,
        4.0,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    assert_eq!(narrow.len(), wide.len());
    wide.equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    assert_eq!(wide.dualize().number_of_elements(), {
        let mut narrow = narrow;
        narrow
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .unwrap();
        narrow.dualize().number_of_elements()
    });
}

#[test]
fn a_wider_cell_indexes_a_depth_a_narrower_one_refuses() {
    let tessellation = sphere(4, 8, 2.0);
    assert_eq!(
        Octree::<u16, usize>::from_features(
            &tessellation,
            1.0e6,
            CurvatureSizing::default(),
            SeparationSizing::default(),
            0
        )
        .err(),
        Some("sizing field exceeds maximum octree depth")
    );
    assert_eq!(u32::length(1 << 20), Some(1u32 << 20));
}

#[test]
fn a_leaner_index_builds_the_same_tree() {
    let tessellation = sphere(4, 8, 2.0);
    let sizing = Sizing::new(
        &tessellation,
        4.0,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    );
    let mut wide = Octree::<u16, usize>::refine(&sizing).unwrap();
    let mut lean = Octree::<u16, u32>::refine(&sizing).unwrap();
    assert_eq!(wide.len(), lean.len());
    wide.equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    lean.equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    assert_eq!(wide.len(), lean.len());
    assert_eq!(
        wide.dualize().number_of_elements(),
        lean.dualize().number_of_elements()
    );
    assert!(size_of::<Node<3, 6, 8, u16, u32, ()>>() < size_of::<Node<3, 6, 8, u16, usize, ()>>());
}

#[test]
fn a_niched_index_builds_the_same_tree() {
    use std::num::NonZeroU32;
    let tessellation = sphere(4, 8, 2.0);
    let sizing = Sizing::new(
        &tessellation,
        4.0,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    );
    let mut wide = Octree::<u16, usize>::refine(&sizing).unwrap();
    let mut lean = Octree::<u16, NonZeroU32>::refine(&sizing).unwrap();
    assert_eq!(wide.len(), lean.len());
    wide.equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    lean.equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    assert_eq!(
        wide.dualize().number_of_elements(),
        lean.dualize().number_of_elements()
    );
    assert!(
        size_of::<Node<3, 6, 8, u16, NonZeroU32, ()>>() < size_of::<Node<3, 6, 8, u16, u32, ()>>()
    );
}

#[test]
fn the_cell_that_set_the_size_field_is_not_a_failure() {
    let tessellation = sphere(4, 8, 2.0);
    (2..=40).for_each(|scale| {
        let scale = scale as Scalar / 2.0;
        assert!(
            Octree::<u16, usize>::from_features(
                &tessellation,
                scale,
                CurvatureSizing::default(),
                SeparationSizing::default(),
                0
            )
            .is_ok(),
            "refused its own finest cell at scale {scale}"
        )
    })
}

/// A unit cube with a stair-step notch of the given depth cut along one top
/// edge (full length in x), closed and manifold. Neither the solid thickness
/// (SDF) nor the surface curvature is affected by the notch - the solid
/// stays bulk-thick on both sides of it, and every face is flat - so it only
/// gets resolved by a size field that can see the narrow gap between the two
/// creases bounding the notch.
fn notched_cube(notch: f64) -> Tessellation {
    let coordinates = vec![
        [0.0, 0.0, 0.0],                 // A
        [1.0, 0.0, 0.0],                 // B
        [1.0, 1.0, 0.0],                 // C
        [0.0, 1.0, 0.0],                 // D
        [0.0, 0.0, 1.0],                 // E
        [1.0, 0.0, 1.0],                 // F
        [0.0, 1.0 - notch, 1.0],         // G
        [1.0, 1.0 - notch, 1.0],         // H
        [0.0, 1.0 - notch, 1.0 - notch], // I
        [1.0, 1.0 - notch, 1.0 - notch], // J
        [0.0, 1.0, 1.0 - notch],         // K
        [1.0, 1.0, 1.0 - notch],         // L
    ];
    let faces = vec![
        [0, 3, 2],
        [0, 2, 1],
        [0, 1, 5],
        [0, 5, 4],
        [4, 5, 7],
        [4, 7, 6],
        [3, 10, 11],
        [3, 11, 2],
        [8, 9, 11],
        [8, 11, 10],
        [8, 6, 7],
        [8, 7, 9],
        [0, 4, 6],
        [0, 6, 8],
        [0, 8, 10],
        [0, 10, 3],
        [1, 2, 11],
        [1, 11, 9],
        [1, 9, 7],
        [1, 7, 5],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

#[test]
fn a_stair_step_notch_is_not_refined_by_thickness_or_curvature_alone() {
    let tessellation = notched_cube(0.05);
    let scale = 5.0;
    let without = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing::default(),
        0,
    )
    .unwrap();
    let with_separation = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing {
            radius: Some(Quantity::new(0.5)),
            hops: 1,
            scale: None,
        },
        0,
    )
    .unwrap();
    assert!(
        with_separation.len() > without.len(),
        "without = {}, with_separation = {}",
        without.len(),
        with_separation.len()
    );
}

fn diagnose(path: &str) {
    let tessellation = Tessellation::try_from(Path::new(path)).unwrap();
    let features = tessellation.features();
    println!(
        "{path}: {} corners, {} creases",
        features.corners().len(),
        features.creases().len()
    );
    let scale = 5.0;
    let mut octree = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing {
            radius: Some(Quantity::new(0.5)),
            hops: 1,
            scale: None,
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let mut sizes: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let nodes: Vec<[f64; 3]> = element
                .iter()
                .map(|&node| {
                    let point = &background.coordinates()[node];
                    [point[0].value(), point[1].value(), point[2].value()]
                })
                .collect();
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            nodes.iter().for_each(|point| {
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            let center: [f64; 3] = from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis]));
            (size, center)
        })
        .collect();
    sizes.sort_by(|(one, _), (two, _)| one.total_cmp(two));
    let smallest = sizes[0].0;
    let largest = sizes[sizes.len() - 1].0;
    let fine: Vec<&(f64, [f64; 3])> = sizes
        .iter()
        .filter(|(size, _)| *size < smallest * 1.5)
        .collect();
    println!(
        "{} elements total, size range [{smallest}, {largest}], {} at the finest size",
        sizes.len(),
        fine.len()
    );
    let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    fine.iter().for_each(|(_, center)| {
        (0..3).for_each(|axis| {
            min[axis] = min[axis].min(center[axis]);
            max[axis] = max[axis].max(center[axis]);
        })
    });
    println!("bounding box of finest-cell centers: {min:?} to {max:?}");
    let mut histogram = std::collections::BTreeMap::<i64, usize>::new();
    sizes.iter().for_each(|(size, _)| {
        *histogram
            .entry((size.log2() * 4.0).round() as i64)
            .or_default() += 1
    });
    histogram.iter().for_each(|(bucket, count)| {
        println!("  size ~2^({:.2}): {count} elements", *bucket as f64 / 4.0)
    });
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_where_the_fine_cells_land() {
    diagnose("/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep.stl");
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_unmatched_creases_on_the_pyramid() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_stairpyramid.stl",
    ))
    .unwrap();
    let features = tessellation.features();
    let creases = features.creases();
    let separation = features.separation(&tessellation, Quantity::new(0.5), 1);
    println!("{} creases total", creases.len());
    let mut unmatched = 0;
    creases
        .iter()
        .zip(separation.iter())
        .for_each(|(segment, entry)| {
            if entry.is_empty() {
                unmatched += 1;
                println!(
                    "UNMATCHED: ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3})",
                    segment[0][0].value(),
                    segment[0][1].value(),
                    segment[0][2].value(),
                    segment[1][0].value(),
                    segment[1][1].value(),
                    segment[1][2].value(),
                );
            }
        });
    println!("{unmatched} unmatched creases out of {}", creases.len());
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_where_the_fine_cells_land_refined() {
    diagnose("/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep_refined.stl");
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_the_stair_pyramid() {
    diagnose("/home/mrbuche/GitHub/autotwin/automesh/cube_stairpyramid.stl");
}

#[test]
#[ignore = "one-off check against a specific file on disk, not a repo fixture"]
fn mesh_the_stair_pyramid() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_stairpyramid.stl",
    ))
    .unwrap();
    for (label, separation, path) in [
        (
            "disabled",
            SeparationSizing::default(),
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairpyramid_disabled.vtu",
        ),
        (
            "enabled",
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: None,
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairpyramid_enabled.vtu",
        ),
    ] {
        let scale = 5.0;
        let mut octree = Octree::<u16, usize>::from_features(
            &tessellation,
            scale,
            CurvatureSizing::default(),
            separation,
            0,
        )
        .unwrap();
        octree
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .unwrap();
        let mut mesh = octree.dualize();
        tessellation.trim(&mut mesh).unwrap();
        let mesh = mesh.buffer(&tessellation, Fitting::Soft).unwrap();
        println!(
            "separation {label:>8}: {} elements, {} nodes",
            mesh.connectivities().iter().flatten().count(),
            mesh.coordinates().len()
        );
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
            Path::new(path),
        ))))
        .unwrap();
    }
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_unmatched_creases_on_the_staircase() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_staircase.stl",
    ))
    .unwrap();
    let features = tessellation.features();
    let creases = features.creases();
    let separation = features.separation(&tessellation, Quantity::new(0.5), 1);
    println!("{} creases total", creases.len());
    let mut unmatched = 0;
    creases
        .iter()
        .zip(separation.iter())
        .enumerate()
        .for_each(|(index, (segment, entry))| {
            if entry.is_empty() {
                unmatched += 1;
                println!(
                    "UNMATCHED crease {index}: ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3})",
                    segment[0][0].value(),
                    segment[0][1].value(),
                    segment[0][2].value(),
                    segment[1][0].value(),
                    segment[1][1].value(),
                    segment[1][2].value(),
                );
            } else {
                println!(
                    "crease {index}: ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3}) has {} partner(s): {}",
                    segment[0][0].value(),
                    segment[0][1].value(),
                    segment[0][2].value(),
                    segment[1][0].value(),
                    segment[1][1].value(),
                    segment[1][2].value(),
                    entry.len(),
                    entry
                        .iter()
                        .map(|s| format!("{}@{:.4}", s.crease, s.distance.value()))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
        });
    println!("{unmatched} unmatched creases out of {}", creases.len());
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_the_staircase() {
    diagnose("/home/mrbuche/GitHub/autotwin/automesh/cube_staircase.stl");
}

#[test]
#[ignore = "one-off check against a specific file on disk, not a repo fixture"]
fn mesh_the_staircase() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_staircase.stl",
    ))
    .unwrap();
    for (label, separation, path) in [
        (
            "disabled",
            SeparationSizing::default(),
            "/home/mrbuche/GitHub/autotwin/automesh/cube_staircase_disabled.vtu",
        ),
        (
            "enabled",
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: None,
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_staircase_enabled.vtu",
        ),
    ] {
        let scale = 5.0;
        let mut octree = Octree::<u16, usize>::from_features(
            &tessellation,
            scale,
            CurvatureSizing::default(),
            separation,
            0,
        )
        .unwrap();
        octree
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .unwrap();
        let mut mesh = octree.dualize();
        tessellation.trim(&mut mesh).unwrap();
        let mesh = mesh.buffer(&tessellation, Fitting::Soft).unwrap();
        println!(
            "separation {label:>8}: {} elements, {} nodes",
            mesh.connectivities().iter().flatten().count(),
            mesh.coordinates().len()
        );
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
            Path::new(path),
        ))))
        .unwrap();
    }
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_unmatched_creases_on_the_heatsink() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink.stl",
    ))
    .unwrap();
    let features = tessellation.features();
    let creases = features.creases();
    let separation = features.separation(&tessellation, Quantity::new(0.5), 1);
    println!("{} creases total", creases.len());
    let mut unmatched = 0;
    creases
        .iter()
        .zip(separation.iter())
        .enumerate()
        .for_each(|(index, (segment, entry))| {
            if entry.is_empty() {
                unmatched += 1;
                println!(
                    "UNMATCHED crease {index}: ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3})",
                    segment[0][0].value(),
                    segment[0][1].value(),
                    segment[0][2].value(),
                    segment[1][0].value(),
                    segment[1][1].value(),
                    segment[1][2].value(),
                );
            } else {
                println!(
                    "crease {index}: ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3}) has {} partner(s): {}",
                    segment[0][0].value(),
                    segment[0][1].value(),
                    segment[0][2].value(),
                    segment[1][0].value(),
                    segment[1][1].value(),
                    segment[1][2].value(),
                    entry.len(),
                    entry
                        .iter()
                        .map(|s| format!("{}@{:.4}", s.crease, s.distance.value()))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
        });
    println!("{unmatched} unmatched creases out of {}", creases.len());
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_heatsink_fine_cells() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink.stl",
    ))
    .unwrap();
    let scale = 5.0;
    let mut octree = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing {
            radius: Some(Quantity::new(0.5)),
            hops: 1,
            scale: None,
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let mut cells: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            element.iter().for_each(|&node| {
                let point = &background.coordinates()[node];
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis].value());
                    max[axis] = max[axis].max(point[axis].value());
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            (
                size,
                from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis])),
            )
        })
        .collect();
    cells.sort_by(|(one, _), (two, _)| one.total_cmp(two));
    let smallest = cells[0].0;
    println!("{} cells, smallest {smallest}", cells.len());
    let fine: Vec<&(f64, [f64; 3])> = cells
        .iter()
        .filter(|(size, _)| *size < smallest * 1.5)
        .collect();
    println!("{} at the finest size", fine.len());
    // Where the finest cells sit, by z band and by y band.
    let mut by_z = std::collections::BTreeMap::<i64, usize>::new();
    let mut by_y = std::collections::BTreeMap::<i64, usize>::new();
    fine.iter().for_each(|(_, center)| {
        *by_z.entry((center[2] * 20.0).floor() as i64).or_default() += 1;
        *by_y.entry((center[1] * 40.0).floor() as i64).or_default() += 1;
    });
    println!("finest cells by z band (width 0.05):");
    by_z.iter().for_each(|(band, count)| {
        println!(
            "  z in [{:.3},{:.3}): {count}",
            *band as f64 / 20.0,
            (*band as f64 + 1.0) / 20.0
        )
    });
    println!("finest cells by y band (width 0.025):");
    by_y.iter().for_each(|(band, count)| {
        println!(
            "  y in [{:.4},{:.4}): {count}",
            *band as f64 / 40.0,
            (*band as f64 + 1.0) / 40.0
        )
    });
    // Inside one slot (y in [0.3, 0.325]), how the finest cells spread in z.
    let slot: Vec<&&(f64, [f64; 3])> = fine
        .iter()
        .filter(|(_, center)| center[1] > 0.3 && center[1] < 0.325)
        .collect();
    let mut slot_z = std::collections::BTreeMap::<i64, usize>::new();
    slot.iter()
        .for_each(|(_, center)| *slot_z.entry((center[2] * 20.0).floor() as i64).or_default() += 1);
    println!(
        "{} finest cells inside the slot y in (0.300,0.325):",
        slot.len()
    );
    slot_z.iter().for_each(|(band, count)| {
        println!(
            "  z in [{:.3},{:.3}): {count}",
            *band as f64 / 20.0,
            (*band as f64 + 1.0) / 20.0
        )
    });
}

#[test]
#[ignore = "one-off diagnostic, not a repo fixture"]
fn diagnose_the_heatsink() {
    diagnose("/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink.stl");
}

#[test]
#[ignore = "one-off check against a specific file on disk, not a repo fixture"]
fn mesh_the_heatsink() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink.stl",
    ))
    .unwrap();
    for (label, separation, path) in [
        (
            "disabled",
            SeparationSizing::default(),
            "/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink_disabled.vtu",
        ),
        (
            "enabled",
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: None,
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_heatsink_enabled.vtu",
        ),
    ] {
        let scale = 5.0;
        let mut octree = Octree::<u16, usize>::from_features(
            &tessellation,
            scale,
            CurvatureSizing::default(),
            separation,
            0,
        )
        .unwrap();
        octree
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .unwrap();
        let mut mesh = octree.dualize();
        tessellation.trim(&mut mesh).unwrap();
        let mesh = mesh.buffer(&tessellation, Fitting::Soft).unwrap();
        println!(
            "separation {label:>8}: {} elements, {} nodes",
            mesh.connectivities().iter().flatten().count(),
            mesh.coordinates().len()
        );
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
            Path::new(path),
        ))))
        .unwrap();
    }
}

#[test]
#[ignore = "one-off check against a specific file on disk, not a repo fixture"]
fn mesh_the_real_stair_step_stl() {
    let tessellation = Tessellation::try_from(Path::new(
        "/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep.stl",
    ))
    .unwrap();
    for (label, scale, separation, path) in [
        (
            "disabled, scale 5",
            5.0,
            SeparationSizing::default(),
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep_scale5.vtu",
        ),
        (
            "enabled, scale 5/5",
            5.0,
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: None,
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep_scale5_separation5.vtu",
        ),
        (
            "enabled, scale 3/5",
            3.0,
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: Some(5.0),
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep_scale3_separation5.vtu",
        ),
        (
            "enabled, scale 4/5",
            4.0,
            SeparationSizing {
                radius: Some(Quantity::new(0.5)),
                hops: 1,
                scale: Some(5.0),
            },
            "/home/mrbuche/GitHub/autotwin/automesh/cube_stairstep_scale4_separation5.vtu",
        ),
    ] {
        // Mirrors automesh's `hexahedralize`: dualize -> trim -> buffer,
        // padding 0 and Weak(1) balancing, the all-hex pathway `mesh hex`
        // actually runs by default (not `cut`, which is only reached via
        // `--element hex-dominant`/`polyhedra`).
        let mut octree = Octree::<u16, usize>::from_features(
            &tessellation,
            scale,
            CurvatureSizing::default(),
            separation,
            0,
        )
        .unwrap();
        octree
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .unwrap();
        let mut mesh = octree.dualize();
        tessellation.trim(&mut mesh).unwrap();
        let mesh = mesh.buffer(&tessellation, Fitting::Soft).unwrap();
        println!(
            "separation {label:>8}: {} elements, {} nodes",
            mesh.connectivities().iter().flatten().count(),
            mesh.coordinates().len()
        );
        mesh.write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
            Path::new(path),
        ))))
        .unwrap();
    }
}

#[test]
fn a_stair_step_notch_only_refines_near_the_notch() {
    // The creases bounding the notch are also vertices of the large, flat
    // triangles covering the rest of the cube, and some of them run the full
    // height of the cube while only their top ends are near anything, so a
    // size field carried on either the triangles or the whole creases would
    // drive the entire model down to the gap size.
    let notch = 0.05;
    let (_, finest) = separation_refinement(&notched_cube(notch));
    // The notch is a full-length slot in x, so only y and z localize.
    finest.iter().for_each(|center| {
        (1..3).for_each(|axis| {
            assert!(
                center[axis] > 1.0 - 4.0 * notch,
                "a finest cell sits at {center:?}"
            )
        })
    });
}

/// Midpoint (one-to-four) subdivision of every triangle, `levels` times.
/// Purely linear, so the shape, its creases, and all of its dihedral angles
/// are untouched - only the triangle count changes.
fn subdivided(tessellation: &Tessellation, levels: usize) -> Tessellation {
    let mut points: Vec<[f64; 3]> = tessellation
        .mesh()
        .coordinates()
        .iter()
        .map(|point| from_fn::<_, 3, _>(|axis| point[axis].value()))
        .collect();
    let mut faces: Vec<[usize; 3]> = tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|element| from_fn::<_, 3, _>(|index| element[index]))
        .collect();
    for _ in 0..levels {
        let mut midpoints = HashMap::<[u64; 3], usize>::new();
        let mut refined = Vec::with_capacity(4 * faces.len());
        faces.iter().for_each(|&[a, b, c]| {
            // Halving a sum of two coordinates is exact and order-independent,
            // so a shared edge yields the same node from either triangle.
            let mut midpoint = |one: usize, two: usize| {
                let point =
                    from_fn::<_, 3, _>(|axis| 0.5 * (points[one][axis] + points[two][axis]));
                *midpoints.entry(point.map(f64::to_bits)).or_insert_with(|| {
                    points.push(point);
                    points.len() - 1
                })
            };
            let (ab, bc, ca) = (midpoint(a, b), midpoint(b, c), midpoint(c, a));
            refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        });
        faces = refined;
    }
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

#[test]
fn narrow_feature_refinement_is_invariant_to_triangulation_density() {
    // Creases are found from dihedral angles, so subdividing flat triangles
    // must not change what is found nor how finely it is resolved. Anything
    // measured in triangle edges rather than in the shape itself - a hop
    // count over individual creases, say - fails here, because a subdivided
    // crease can then pair its own collinear pieces with each other.
    let coarse = notched_cube(0.05);
    let fine = subdivided(&coarse, 2);
    assert_eq!(
        16 * coarse.mesh().connectivities().iter().flatten().count(),
        fine.mesh().connectivities().iter().flatten().count()
    );
    let radius = Quantity::new(0.5);
    let gap = |tessellation: &Tessellation| {
        tessellation
            .features()
            .separation(tessellation, radius, 1)
            .iter()
            .flatten()
            .map(|separation| separation.distance.value())
            .fold(f64::INFINITY, f64::min)
    };
    assert!((gap(&coarse) - gap(&fine)).abs() < 1.0e-12);
    let (coarse_size, coarse_finest) = separation_refinement(&coarse);
    let (fine_size, fine_finest) = separation_refinement(&fine);
    assert!((coarse_size - fine_size).abs() < 1.0e-12);
    let extent = |finest: &[[f64; 3]]| {
        (1..3)
            .map(|axis| {
                let low = finest
                    .iter()
                    .map(|center| center[axis])
                    .fold(f64::INFINITY, f64::min);
                let high = finest
                    .iter()
                    .map(|center| center[axis])
                    .fold(f64::NEG_INFINITY, f64::max);
                high - low
            })
            .fold(0.0_f64, f64::max)
    };
    assert!((extent(&coarse_finest) - extent(&fine_finest)).abs() < 1.0e-12);
}

/// The finest cell size the octree refines a tessellation to, and where the
/// centers of the cells at that size sit, with separation-driven refinement
/// enabled.
fn separation_refinement(tessellation: &Tessellation) -> (f64, Vec<[f64; 3]>) {
    let mut octree = Octree::<u16, usize>::from_features(
        tessellation,
        5.0,
        CurvatureSizing::default(),
        SeparationSizing {
            radius: Some(Quantity::new(0.5)),
            hops: 1,
            scale: None,
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let cells: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            element.iter().for_each(|&node| {
                let point = &background.coordinates()[node];
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis].value());
                    max[axis] = max[axis].max(point[axis].value());
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            (
                size,
                from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis])),
            )
        })
        .collect();
    let smallest = cells
        .iter()
        .map(|&(size, _)| size)
        .fold(f64::INFINITY, f64::min);
    (
        smallest,
        cells
            .iter()
            .filter(|&&(size, _)| size < smallest * 1.5)
            .map(|&(_, center)| center)
            .collect(),
    )
}

/// A stair-stepped pyramid: a `1 x 1 x base` slab carrying `tiers` further
/// steps, each `step` tall and inset `step` from the one below. Every tier
/// leaves a tread only `step` wide running all the way around all four
/// sides, so each crease bounding one has several others exactly `step`
/// away - the one across the tread, the one across the riser, and the ones
/// capping the tread at each corner - rather than a single obvious partner.
fn stair_pyramid(base: f64, step: f64, tiers: usize) -> Tessellation {
    let mut triangles: Vec<[[f64; 3]; 3]> = Vec::new();
    let mut quad = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
        triangles.push([a, b, c]);
        triangles.push([a, c, d])
    };
    // Level zero is the slab; level `k` is inset `k` steps and sits on top of
    // level `k - 1`. Rings run counter-clockwise seen from above, so walking
    // one puts the outside on the right and every face comes out outward.
    let height = |level: usize| match level {
        0 => 0.0,
        _ => base + step * (level - 1) as f64,
    };
    let ring = |level: usize| -> [[f64; 2]; 4] {
        let (low, high) = (step * level as f64, 1.0 - step * level as f64);
        [[low, low], [high, low], [high, high], [low, high]]
    };
    let at = |[x, y]: [f64; 2], z: f64| [x, y, z];
    let base_ring = ring(0);
    quad(
        at(base_ring[0], 0.0),
        at(base_ring[3], 0.0),
        at(base_ring[2], 0.0),
        at(base_ring[1], 0.0),
    );
    for level in 0..=tiers {
        let (ring, low, high) = (ring(level), height(level), height(level + 1));
        for edge in 0..4 {
            let (one, two) = (ring[edge], ring[(edge + 1) % 4]);
            quad(at(one, low), at(two, low), at(two, high), at(one, high))
        }
    }
    for level in 1..=tiers {
        let (outer, inner, tread) = (ring(level - 1), ring(level), height(level));
        for edge in 0..4 {
            quad(
                at(outer[edge], tread),
                at(outer[(edge + 1) % 4], tread),
                at(inner[(edge + 1) % 4], tread),
                at(inner[edge], tread),
            )
        }
    }
    let top = ring(tiers);
    let peak = height(tiers + 1);
    quad(
        at(top[0], peak),
        at(top[1], peak),
        at(top[2], peak),
        at(top[3], peak),
    );
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut lookup = HashMap::<[u64; 3], usize>::new();
    let faces: Vec<[usize; 3]> = triangles
        .iter()
        .map(|triangle| {
            from_fn::<_, 3, _>(|corner| {
                let point = triangle[corner];
                *lookup.entry(point.map(f64::to_bits)).or_insert_with(|| {
                    points.push(point);
                    points.len() - 1
                })
            })
        })
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

#[test]
fn every_narrow_stretch_of_every_crease_is_refined() {
    let pyramid = stair_pyramid(0.5, 0.1, 3);
    coverage(&pyramid, &pyramid);
}

#[test]
fn narrow_stretch_coverage_survives_subdivision() {
    // Subdividing the input must not move any of it: the same stretches of
    // the same shape have to come out refined the same way, so the shape is
    // measured once, on the coarse mesh, and asked of both.
    let pyramid = stair_pyramid(0.5, 0.1, 3);
    coverage(&pyramid, &subdivided(&pyramid, 1));
}

/// Checks that wherever `shape` runs close to itself, `meshed` - the same
/// shape, however it happens to be triangulated - is refined to match.
fn coverage(shape: &Tessellation, meshed: &Tessellation) {
    // A crease usually bounds a narrow feature along only part of its length,
    // and usually has several partners tied for nearest - the ones capping
    // the feature at either end as well as the one running alongside it. Keep
    // only one of them and whichever stretch the others covered goes coarse,
    // which shows up as steps that look skipped rather than as anything the
    // crease matching itself reports as missing.
    let step = 0.1;
    let scale = 5.0;
    let creases = shape.features().creases();
    let nodes = shape.features().crease_nodes();
    let (finest, cells) = cell_sizes(meshed, scale);
    creases.iter().enumerate().for_each(|(this, segment)| {
        (0..=10).for_each(|piece| {
            let fraction = piece as f64 / 10.0;
            let point = from_fn::<_, 3, _>(|axis| {
                segment[0][axis].value() * (1.0 - fraction) + segment[1][axis].value() * fraction
            });
            // How near this stretch of the crease comes to any crease that is
            // not one of its own neighbors, measured independently of what
            // `separation` chose to report.
            let gap = creases
                .iter()
                .enumerate()
                .filter(|&(other, _)| {
                    other != this && nodes[other].iter().all(|node| !nodes[this].contains(node))
                })
                .map(|(_, other)| distance_to(&point, other))
                .fold(f64::INFINITY, f64::min);
            if gap < 2.0 * step {
                let size = cells(point);
                assert!(
                    size <= 1.5 * gap / scale,
                    "the crease at {point:?} runs {gap} from another but sits in cells of {size}"
                );
                assert!(
                    size >= finest,
                    "{size} is finer than the finest cell {finest}"
                )
            }
        })
    });
}

fn distance_to(point: &[f64; 3], segment: &[crate::geometry::Coordinate<3>; 2]) -> f64 {
    let along = from_fn::<_, 3, _>(|axis| segment[1][axis].value() - segment[0][axis].value());
    let length: f64 = along.iter().map(|entry| entry * entry).sum();
    let offset = from_fn::<_, 3, _>(|axis| point[axis] - segment[0][axis].value());
    let fraction = match length > 0.0 {
        true => {
            ((0..3).map(|axis| offset[axis] * along[axis]).sum::<f64>() / length).clamp(0.0, 1.0)
        }
        false => 0.0,
    };
    (0..3)
        .map(|axis| (offset[axis] - fraction * along[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The finest cell the octree refines a tessellation to, and a lookup of the
/// finest cell near any point.
fn cell_sizes(tessellation: &Tessellation, scale: f64) -> (f64, impl Fn([f64; 3]) -> f64) {
    let mut octree = Octree::<u16, usize>::from_features(
        tessellation,
        scale,
        CurvatureSizing::default(),
        SeparationSizing {
            radius: Some(Quantity::new(0.5)),
            hops: 1,
            scale: None,
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let cells: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            element.iter().for_each(|&node| {
                let point = &background.coordinates()[node];
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis].value());
                    max[axis] = max[axis].max(point[axis].value());
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            (
                size,
                from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis])),
            )
        })
        .collect();
    let finest = cells
        .iter()
        .map(|&(size, _)| size)
        .fold(f64::INFINITY, f64::min);
    let bin = 4.0 * finest;
    let mut grid = HashMap::<[i64; 3], f64>::new();
    cells.iter().for_each(|&(size, center)| {
        let key = from_fn::<_, 3, _>(|axis| (center[axis] / bin).floor() as i64);
        let slot = grid.entry(key).or_insert(f64::INFINITY);
        *slot = slot.min(size)
    });
    (finest, move |point: [f64; 3]| {
        let key = from_fn::<_, 3, _>(|axis| (point[axis] / bin).floor() as i64);
        (-1..=1)
            .flat_map(|i| (-1..=1).flat_map(move |j| (-1..=1).map(move |k| [i, j, k])))
            .filter_map(|[i, j, k]| grid.get(&[key[0] + i, key[1] + j, key[2] + k]))
            .fold(f64::INFINITY, |best, &size| best.min(size))
    })
}
