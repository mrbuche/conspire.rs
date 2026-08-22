use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, Tessellation, test::sphere},
        ntree::node::{Node, cell::Cell},
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing, Sizing},
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::{array::from_fn, collections::HashMap};

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
    let loose =
        Octree::<u16, usize>::from_features(&tessellation, scale, curvature(Quantity::new(1.0)), 0)
            .unwrap();
    let medium = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-2)),
        0,
    )
    .unwrap();
    let tight = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-3)),
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
        0,
    )
    .unwrap();
    let with_default =
        Octree::<u16, usize>::from_features(&tessellation, scale, CurvatureSizing::default(), 0)
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
    let mut octree = Octree::<u16, usize>::from_features(tessellation, 5.0, curvature, 0).unwrap();
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
        Octree::<u16, usize>::from_features(&tessellation, 1.0e6, CurvatureSizing::default(), 0)
            .err(),
        Some("sizing field exceeds maximum octree depth")
    );
    assert!(
        Octree::<u16, usize>::from_features(&tessellation, 4.0, CurvatureSizing::default(), 0)
            .is_ok()
    );
}

#[test]
fn a_wider_cell_carries_the_whole_pipeline() {
    let tessellation = sphere(4, 8, 2.0);
    let narrow =
        Octree::<u16, usize>::from_features(&tessellation, 4.0, CurvatureSizing::default(), 0)
            .unwrap();
    let mut wide =
        Octree::<u32, usize>::from_features(&tessellation, 4.0, CurvatureSizing::default(), 0)
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
        Octree::<u16, usize>::from_features(&tessellation, 1.0e6, CurvatureSizing::default(), 0)
            .err(),
        Some("sizing field exceeds maximum octree depth")
    );
    assert_eq!(u32::length(1 << 20), Some(1u32 << 20));
}

#[test]
fn a_leaner_index_builds_the_same_tree() {
    let tessellation = sphere(4, 8, 2.0);
    let sizing = Sizing::new(&tessellation, 4.0, CurvatureSizing::default(), 0);
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
    let sizing = Sizing::new(&tessellation, 4.0, CurvatureSizing::default(), 0);
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
                0
            )
            .is_ok(),
            "refused its own finest cell at scale {scale}"
        )
    })
}
