use crate::geometry::mesh::tessellation::NORMALIZED_DIRECTIONS;
use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Interfaces, Tessellation, Watertight,
            test::{cuboid, solid},
        },
        solid::SolidOracle,
    },
    math::{
        CrossProduct, Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
};
use core::array::from_fn;

type Box3 = ([i64; 3], [i64; 3]);

fn make(boxes: &[Box3]) -> Vec<Watertight> {
    boxes
        .iter()
        .map(|(low, high)| solid(low.map(|x| x as f64 * 0.25), high.map(|x| x as f64 * 0.25)))
        .collect()
}

fn contains(boxed: &Box3, cell: [i64; 3]) -> bool {
    (0..3).all(|axis| boxed.0[axis] <= cell[axis] && cell[axis] < boxed.1[axis])
}

fn owned(boxes: &[Box3], material: usize) -> f64 {
    let mut cells = 0;
    for x in -8..16 {
        for y in -8..16 {
            for z in -8..16 {
                let cell = [x, y, z];
                let owner = boxes.iter().rposition(|boxed| contains(boxed, cell));
                cells += (owner == Some(material)) as usize;
            }
        }
    }
    cells as f64 / 64.0
}

fn assert_materials(boxes: &[Box3], touching: bool) -> Result<(), AssertionError> {
    let interfaces = Interfaces::new(&make(boxes)).unwrap();
    assert_eq!(interfaces.inside().len(), interfaces.outside().len());
    for material in 0..boxes.len() {
        let expected = owned(boxes, material);
        match interfaces.boundary(material) {
            Ok(boundary) => {
                let defects = boundary.defects();
                if touching {
                    assert!(defects.boundary_edges.is_empty(), "{material}: {defects:?}");
                    assert!(defects.self_intersections.is_empty() && !defects.inward);
                } else {
                    assert!(defects.is_empty(), "{material}: {defects:?}");
                }
                Assert::default()
                    .eq_within_tols(boundary.enclosed_volume(), &Quantity::new(expected))?;
            }
            Err(_) => assert_eq!(expected, 0.0, "material {material}"),
        }
    }
    Ok(())
}

#[test]
fn a_host_with_two_separate_solids_inside() -> Result<(), AssertionError> {
    assert_materials(
        &[
            ([0, 0, 0], [8, 8, 8]),
            ([1, 1, 1], [3, 3, 3]),
            ([5, 5, 5], [7, 7, 7]),
        ],
        false,
    )
}

#[test]
fn a_host_with_two_separate_solids_through_its_surface() -> Result<(), AssertionError> {
    assert_materials(
        &[
            ([0, 0, 0], [8, 8, 8]),
            ([-2, 1, 1], [3, 3, 3]),
            ([5, 5, 5], [10, 7, 7]),
        ],
        false,
    )
}

#[test]
fn a_chain_of_overlaps_where_the_ends_do_not_meet() -> Result<(), AssertionError> {
    assert_materials(
        &[
            ([0, 0, 0], [4, 4, 4]),
            ([3, 1, 1], [7, 5, 5]),
            ([6, 2, 2], [10, 6, 6]),
        ],
        false,
    )
}

#[test]
fn three_solids_side_by_side_sharing_faces() -> Result<(), AssertionError> {
    assert_materials(
        &[
            ([0, 0, 0], [4, 4, 4]),
            ([4, 0, 0], [8, 4, 4]),
            ([8, 0, 0], [12, 4, 4]),
        ],
        false,
    )
}

#[test]
fn a_solid_drawn_over_another_replaces_it_where_they_overlap() -> Result<(), AssertionError> {
    let boxes = [([0, 0, 0], [4, 4, 4]), ([2, 2, 2], [6, 6, 6])];
    assert_materials(&boxes, false)?;
    assert_eq!(owned(&boxes, 1), 1.0);
    assert!(owned(&boxes, 0) < 1.0);
    let interfaces = Interfaces::new(&make(&boxes)).unwrap();
    assert!(interfaces.outside().contains(&Some(0)));
    assert!(interfaces.outside().iter().any(|&o| o.is_none()));
    Ok(())
}

#[test]
fn the_same_solids_in_the_other_order_swap_the_notch() -> Result<(), AssertionError> {
    assert_materials(&[([2, 2, 2], [6, 6, 6]), ([0, 0, 0], [4, 4, 4])], false)
}

#[test]
fn a_surface_across_coplanar_faces_cuts_them_along_the_edges_of_each() -> Result<(), AssertionError>
{
    for boxes in [
        [
            ([3, 5, 2], [8, 8, 4]),
            ([1, 7, 1], [6, 10, 3]),
            ([4, 7, 3], [8, 10, 7]),
        ],
        [
            ([2, 5, 5], [7, 7, 7]),
            ([3, 6, 5], [5, 9, 8]),
            ([1, 4, 6], [5, 9, 7]),
        ],
        [
            ([3, 5, 2], [4, 8, 5]),
            ([0, 7, 4], [4, 11, 9]),
            ([1, 7, 5], [5, 12, 10]),
        ],
        [
            ([3, 3, 6], [4, 5, 9]),
            ([2, 1, 2], [5, 6, 6]),
            ([3, 4, 2], [6, 5, 7]),
        ],
        [
            ([6, 6, 7], [11, 10, 10]),
            ([6, 7, 5], [11, 8, 9]),
            ([6, 6, 4], [10, 10, 9]),
        ],
        [
            ([3, 0, 3], [4, 3, 8]),
            ([1, 0, 1], [5, 2, 6]),
            ([3, 0, 3], [7, 2, 7]),
        ],
        [
            ([2, 5, 1], [6, 8, 6]),
            ([6, 2, 1], [11, 6, 4]),
            ([2, 2, 1], [6, 6, 6]),
        ],
    ] {
        assert_materials(&boxes, true)?;
    }
    Ok(())
}

#[test]
fn three_surfaces_meeting_on_the_diagonals_of_their_faces() -> Result<(), AssertionError> {
    assert_materials(
        &[
            ([0, 0, 0], [4, 4, 4]),
            ([2, 2, 0], [6, 6, 4]),
            ([2, 0, 2], [6, 4, 6]),
        ],
        true,
    )
}

const TRIPLE_POINT: [&str; 3] = [
    "two segments cross",
    "two triple points are the same point",
    "a triple point is too near the end of a curve to be put in order",
];

#[test]
fn random_boxes_on_a_coarse_grid_give_closed_materials_or_an_error() -> Result<(), AssertionError> {
    let mut state = 17u64;
    let mut next = |n: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % n
    };
    let mut done = 0;
    for _ in 0..300 {
        let boxes: Vec<Box3> = (0..3)
            .map(|_| {
                let low: [i64; 3] = core::array::from_fn(|_| next(8) as i64);
                let high = core::array::from_fn(|axis| low[axis] + 1 + next(5) as i64);
                (low, high)
            })
            .collect();
        match Interfaces::new(&make(&boxes)) {
            Ok(_) => {
                assert_materials(&boxes, true)?;
                done += 1;
            }
            Err(message) => panic!("{message} {boxes:?}"),
        }
    }
    assert_eq!(done, 300);
    Ok(())
}

fn tilted(low: [f64; 3], high: [f64; 3], angles: [f64; 3]) -> Watertight {
    let cube = cuboid(low, high);
    let rotate = |point: [f64; 3]| {
        let mut point = point;
        for (axis, angle) in angles.iter().enumerate() {
            let (sin, cos) = angle.sin_cos();
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            (point[u], point[v]) = (
                cos * point[u] - sin * point[v],
                sin * point[u] + cos * point[v],
            );
        }
        point
    };
    let vertices: Vec<Coordinate<3>> = cube
        .mesh()
        .coordinates()
        .iter()
        .map(|p| rotate(from_fn(|axis| p[axis].value())).into())
        .collect();
    let faces: Vec<[usize; 3]> = cube
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|nodes| [nodes[0], nodes[1], nodes[2]])
        .collect();
    Watertight::new((faces, vertices).into()).unwrap()
}

fn assert_sampled(solids: &[Watertight], samples: usize) {
    let interfaces = Interfaces::new(solids).unwrap();
    let boundaries: Vec<Option<Tessellation>> = (0..solids.len())
        .map(|material| {
            interfaces.boundary(material).ok().inspect(|boundary| {
                let defects = boundary.defects();
                assert!(defects.boundary_edges.is_empty(), "{defects:?}");
                assert!(defects.non_manifold_edges.is_empty(), "{defects:?}");
                assert!(defects.inconsistent_edges.is_empty(), "{defects:?}");
                assert!(defects.self_intersections.is_empty() && !defects.inward);
            })
        })
        .collect();
    let encloses = |boundary: &Tessellation, point: &Coordinate<3>| {
        let mesh = boundary.mesh();
        let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
        let normals = boundary.normals().iter().flatten().collect();
        boundary.encloses(
            point,
            mesh.coordinates(),
            &elements,
            &normals,
            &NORMALIZED_DIRECTIONS,
        )
    };
    let mut state = 5u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut checked = 0;
    for _ in 0..samples {
        let point = from_fn(|_| -1.0 + 9.0 * next()).into();
        let distances: Vec<f64> = solids
            .iter()
            .map(|solid| solid.signed_distance(&point).value())
            .collect();
        if distances.iter().any(|d| d.abs() < 1e-6) {
            continue;
        }
        let owner = distances.iter().rposition(|&d| d < 0.0);
        for (material, boundary) in boundaries.iter().enumerate() {
            let inside = boundary
                .as_ref()
                .is_some_and(|boundary| encloses(boundary, &point));
            assert_eq!(inside, owner == Some(material), "{point:?} {material}");
        }
        checked += 1;
    }
    assert!(checked > 3 * samples / 4);
}

#[test]
fn three_tilted_solids_meeting_at_points_give_the_materials_they_should() {
    let solids = [
        tilted([0.0; 3], [4.0; 3], [0.0, 0.0, 0.1]),
        tilted([2.0, 2.0, 0.0], [6.0, 6.0, 4.0], [0.13, 0.0, 0.0]),
        tilted([2.0, 0.0, 2.0], [6.0, 4.0, 6.0], [0.0, 0.17, 0.0]),
    ];
    assert_sampled(&solids, 4000);
}

#[test]
fn random_tilted_triples_give_the_materials_they_should_or_an_error() {
    let mut state = 29u64;
    let mut next = |scale: f64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        scale * ((state >> 11) as f64 / (1u64 << 53) as f64)
    };
    let (mut done, mut failed) = (0, 0);
    for _ in 0..40 {
        let solids: Vec<Watertight> = (0..3)
            .map(|_| {
                let low: [f64; 3] = from_fn(|_| next(3.0));
                let high = from_fn(|axis| low[axis] + 1.5 + next(3.0));
                tilted(low, high, from_fn(|_| next(0.6) - 0.3))
            })
            .collect();
        match Interfaces::new(&solids) {
            Ok(_) => {
                assert_sampled(&solids, 600);
                done += 1;
            }
            Err(message) => {
                assert!(TRIPLE_POINT.contains(&message), "{message}");
                failed += 1;
            }
        }
    }
    assert!(done > 25, "{done} {failed}");
}

fn tetrahedron(corners: [[f64; 3]; 4]) -> Watertight {
    let [a, b, c, d] = corners.map(Coordinate::from);
    let handed = ((&b - &a).cross(&(&c - &a)) * &(&d - &a)).value();
    let (b, c) = if handed > 0.0 { (b, c) } else { (c, b) };
    let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    Watertight::new((faces, vec![a, b, c, d]).into()).unwrap()
}

#[test]
fn a_slab_across_the_curve_of_two_solids_makes_two_points_on_it() {
    let solids = [
        tilted([0.0; 3], [4.0; 3], [0.0, 0.0, 0.03]),
        tilted([1.5, 1.0, 0.0], [5.5, 5.0, 4.0], [0.04, 0.0, 0.0]),
        tilted([3.0, 0.0, 2.7], [6.0, 3.0, 3.7], [0.0, 0.03, 0.05]),
    ];
    assert_sampled(&solids, 2000);
}

#[test]
fn four_solids_with_a_face_through_one_point_meet_there() {
    let solids = [
        tetrahedron([
            [-1.0, -1.0, 0.0],
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, 0.0],
            [0.513, 0.487, 2.071],
        ]),
        tetrahedron([
            [0.0, -1.0, -1.0],
            [0.0, 3.0, -1.0],
            [0.0, -1.0, 3.0],
            [-2.063, 0.519, 0.481],
        ]),
        tetrahedron([
            [-1.0, 0.0, -1.0],
            [-1.0, 0.0, 3.0],
            [3.0, 0.0, -1.0],
            [0.472, -2.041, 0.557],
        ]),
        tetrahedron([
            [-1.0, -1.0, 2.0],
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0],
            [1.517, 1.463, 1.529],
        ]),
    ];
    assert_sampled(&solids, 3000);
}
