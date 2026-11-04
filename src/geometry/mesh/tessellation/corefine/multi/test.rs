use crate::{
    geometry::mesh::{Interfaces, Watertight, test::solid},
    math::Quantity,
    units::Volume,
};

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

fn close(found: Quantity<Volume>, expected: f64) {
    assert!(
        (found - Quantity::new(expected)).abs() < Quantity::new(1e-12),
        "{found} != {expected}"
    );
}

fn assert_materials(boxes: &[Box3], touching: bool) {
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
                close(boundary.enclosed_volume(), expected);
            }
            Err(_) => assert_eq!(expected, 0.0, "material {material}"),
        }
    }
}

#[test]
fn a_host_with_two_separate_solids_inside() {
    assert_materials(
        &[
            ([0, 0, 0], [8, 8, 8]),
            ([1, 1, 1], [3, 3, 3]),
            ([5, 5, 5], [7, 7, 7]),
        ],
        false,
    );
}

#[test]
fn a_host_with_two_separate_solids_through_its_surface() {
    assert_materials(
        &[
            ([0, 0, 0], [8, 8, 8]),
            ([-2, 1, 1], [3, 3, 3]),
            ([5, 5, 5], [10, 7, 7]),
        ],
        false,
    );
}

#[test]
fn a_chain_of_overlaps_where_the_ends_do_not_meet() {
    assert_materials(
        &[
            ([0, 0, 0], [4, 4, 4]),
            ([3, 1, 1], [7, 5, 5]),
            ([6, 2, 2], [10, 6, 6]),
        ],
        false,
    );
}

#[test]
fn three_solids_side_by_side_sharing_faces() {
    assert_materials(
        &[
            ([0, 0, 0], [4, 4, 4]),
            ([4, 0, 0], [8, 4, 4]),
            ([8, 0, 0], [12, 4, 4]),
        ],
        false,
    );
}

#[test]
fn a_solid_drawn_over_another_replaces_it_where_they_overlap() {
    let boxes = [([0, 0, 0], [4, 4, 4]), ([2, 2, 2], [6, 6, 6])];
    assert_materials(&boxes, false);
    assert_eq!(owned(&boxes, 1), 1.0);
    assert!(owned(&boxes, 0) < 1.0);
    let interfaces = Interfaces::new(&make(&boxes)).unwrap();
    assert!(interfaces.outside().contains(&Some(0)));
    assert!(interfaces.outside().iter().any(|&o| o.is_none()));
}

#[test]
fn the_same_solids_in_the_other_order_swap_the_notch() {
    assert_materials(&[([2, 2, 2], [6, 6, 6]), ([0, 0, 0], [4, 4, 4])], false);
}

#[test]
fn three_surfaces_meeting_at_a_point_are_an_error() {
    let solids = make(&[
        ([0, 0, 0], [4, 4, 4]),
        ([2, 2, 0], [6, 6, 4]),
        ([2, 0, 2], [6, 4, 6]),
    ]);
    let message = Interfaces::new(&solids).err().unwrap();
    assert!(TRIPLE_POINT.contains(&message), "{message}");
}

const TRIPLE_POINT: [&str; 2] = [
    "two segments cross",
    "two points on an edge are the same point",
];

#[test]
fn random_boxes_on_a_coarse_grid_give_closed_materials_or_an_error() {
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
                let low: [i64; 3] = std::array::from_fn(|_| next(8) as i64);
                let high = std::array::from_fn(|axis| low[axis] + 1 + next(5) as i64);
                (low, high)
            })
            .collect();
        match Interfaces::new(&make(&boxes)) {
            Ok(_) => {
                assert_materials(&boxes, true);
                done += 1;
            }
            Err(message) => assert!(TRIPLE_POINT.contains(&message), "{message}"),
        }
    }
    assert!(done > 200, "{done}");
}
