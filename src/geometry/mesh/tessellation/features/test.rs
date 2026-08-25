use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, tessellation::Tessellation},
    },
    math::{Quantity, Tensor, TensorVec},
};
use std::collections::HashMap;

fn cube(minimum: [f64; 3], maximum: [f64; 3]) -> Tessellation {
    let [x0, y0, z0] = minimum;
    let [x1, y1, z1] = maximum;
    let coordinates = vec![
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ];
    let quads: [[usize; 4]; 6] = [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ];
    let faces: Vec<[usize; 3]> = quads
        .iter()
        .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn sphere(refinements: usize) -> Tessellation {
    let mut coordinates = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut faces = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    (0..refinements).for_each(|_| {
        let mut cache = HashMap::new();
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let mut midpoint = |one: usize, two: usize| {
                    let key = if one < two { [one, two] } else { [two, one] };
                    *cache.entry(key).or_insert_with(|| {
                        let point: Vec<f64> = (0..3)
                            .map(|d| 0.5 * (coordinates[one][d] + coordinates[two][d]))
                            .collect();
                        let norm = point.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
                        coordinates.push([point[0] / norm, point[1] / norm, point[2] / norm]);
                        coordinates.len() - 1
                    })
                };
                let ab = midpoint(a, b);
                let bc = midpoint(b, c);
                let ca = midpoint(c, a);
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect()
    });
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

#[test]
fn a_cube_has_eight_corners_and_twelve_creases() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let features = tessellation.features();
    assert_eq!(features.corners().len(), 8);
    assert_eq!(features.creases().len(), 12);
}

#[test]
fn the_corners_of_a_cube_are_its_vertices() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let features = tessellation.features();
    features.corners().iter().for_each(|corner| {
        (0..3).for_each(|d| {
            assert!(
                corner[d] == Quantity::new(0.0) || corner[d] == Quantity::new(1.0),
                "{corner}"
            )
        })
    });
}

#[test]
fn a_sphere_has_no_features() {
    let tessellation = sphere(3);
    let features = tessellation.features();
    assert!(features.corners().is_empty());
    assert!(features.creases().is_empty());
}

#[test]
fn a_corner_is_found_only_within_the_radius() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(Quantity::new(0.2));
    let point = Coordinate::const_from([0.1, 0.1, 0.1]);
    let (corner, distance) = index.nearest_corner(&point, Quantity::new(0.2)).unwrap();
    assert_eq!(index.corner(corner), &Coordinate::const_from([0.0; 3]));
    assert!((distance.value() - 3.0_f64.sqrt() * 0.1).abs() < 1.0e-12);
    assert!(index.nearest_corner(&point, Quantity::new(0.1)).is_none());
}

#[test]
fn a_crease_is_found_at_its_closest_point() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(Quantity::new(0.3));
    let point = Coordinate::const_from([0.05, 0.05, 0.5]);
    let closest = index.nearest_crease(&point, Quantity::new(0.3)).unwrap();
    assert!(
        (&closest - &Coordinate::const_from([0.0, 0.0, 0.5])).norm() < Quantity::new(1.0e-12),
        "{closest}"
    );
}

/// Three flat panels folded like a stair step: top -> riser -> tread, with a
/// gap of `notch` between the outer edges of the top and tread panels. This
/// is not watertight (only used to exercise crease detection and
/// separation), matching how `triangles_that_share_no_nodes_have_no_features`
/// already relies on open connectivity being fine for creases.
fn stair_step(notch: f64) -> Tessellation {
    let coordinates = vec![
        // top panel, z = 1, y: 0 to 1 - notch
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0 - notch, 1.0],
        [0.0, 1.0 - notch, 1.0],
        // riser panel, y = 1 - notch, z: 1 - notch to 1
        [0.0, 1.0 - notch, 1.0 - notch],
        [1.0, 1.0 - notch, 1.0 - notch],
        // tread panel, z = 1 - notch, y: 1 - notch to 1
        [0.0, 1.0, 1.0 - notch],
        [1.0, 1.0, 1.0 - notch],
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [3, 2, 5],
        [3, 5, 4],
        [4, 5, 7],
        [4, 7, 6],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

#[test]
fn a_stair_step_has_two_creases() {
    let tessellation = stair_step(0.1);
    let features = tessellation.features();
    assert_eq!(features.creases().len(), 2);
}

#[test]
fn a_stair_step_notch_is_seen_as_a_narrow_feature() {
    let tessellation = stair_step(0.1);
    let features = tessellation.features();
    let separation = features.separation(&tessellation, Quantity::new(1.0), 1);
    // the two creases are not adjacent in the crease graph (they share no
    // node) but sit 0.1 apart in space, across the riser panel.
    separation.iter().for_each(|entry| {
        let distance = entry.first().expect("no nearby crease").distance;
        assert!(
            (distance.value() - 0.1).abs() < 1.0e-12,
            "{}",
            distance.value()
        )
    });
}

#[test]
fn a_wider_stair_step_notch_reports_a_larger_separation() {
    let tessellation = stair_step(0.3);
    let features = tessellation.features();
    let separation = features.separation(&tessellation, Quantity::new(1.0), 1);
    separation.iter().for_each(|entry| {
        let distance = entry.first().expect("no nearby crease").distance;
        assert!(
            (distance.value() - 0.3).abs() < 1.0e-12,
            "{}",
            distance.value()
        )
    });
}

#[test]
fn a_stair_step_notch_beyond_the_radius_is_not_seen() {
    let tessellation = stair_step(0.1);
    let features = tessellation.features();
    let separation = features.separation(&tessellation, Quantity::new(0.05), 1);
    separation
        .iter()
        .for_each(|entry| assert!(entry.is_empty(), "{entry:?}"));
}

/// A slot of width `gap` and depth `depth`: a floor panel with a wall rising
/// from either edge of it, each wall running out into a flat top panel. The
/// two creases where the walls meet the floor bound a narrow ribbon of
/// surface - the slot floor - while the two where they meet the top panels
/// are exactly as close together with nothing at all between them.
fn slot(gap: f64, depth: f64) -> Tessellation {
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, gap, 0.0],
        [0.0, gap, 0.0],
        [0.0, 0.0, depth],
        [1.0, 0.0, depth],
        [0.0, gap, depth],
        [1.0, gap, depth],
        [0.0, -1.0, depth],
        [1.0, -1.0, depth],
        [0.0, gap + 1.0, depth],
        [1.0, gap + 1.0, depth],
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 4, 5],
        [0, 5, 1],
        [3, 2, 7],
        [3, 7, 6],
        [4, 8, 9],
        [4, 9, 5],
        [6, 7, 11],
        [6, 11, 10],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

#[test]
fn only_the_creases_with_surface_between_them_bound_a_slot() {
    // The mouth of a slot is not a narrow feature: the two edges either side
    // of it run a gap apart just as the two either side of its floor do, but
    // nothing lies between them, so nothing there needs resolving to the gap.
    let tessellation = slot(0.1, 1.0);
    let features = tessellation.features();
    assert_eq!(features.creases().len(), 4);
    let separation = features.separation(&tessellation, Quantity::new(0.5), 1);
    let floor = features
        .creases()
        .iter()
        .zip(separation.iter())
        .filter(|(segment, _)| segment[0][2] == Quantity::new(0.0))
        .count();
    assert_eq!(floor, 2);
    features
        .creases()
        .iter()
        .zip(separation.iter())
        .for_each(|(segment, entry)| {
            if segment[0][2] == Quantity::new(0.0) {
                let distance = entry.first().expect("the slot floor went unseen").distance;
                assert!((distance.value() - 0.1).abs() < 1.0e-12, "{distance}")
            } else {
                assert!(entry.is_empty(), "the slot mouth was seen as narrow")
            }
        })
}

#[test]
fn without_graph_exclusion_a_cube_corner_looks_like_a_narrow_feature() {
    // three creases meet at every corner of a cube, touching at distance
    // zero; with no hops excluded beyond the crease itself, that shared
    // corner would be misread as an arbitrarily narrow feature.
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let features = tessellation.features();
    let separation = features.separation(&tessellation, Quantity::new(0.5), 0);
    assert!(
        separation.iter().all(|entry| {
            entry
                .first()
                .is_some_and(|entry| entry.distance == Quantity::new(0.0))
        }),
        "{separation:?}"
    );
}

#[test]
fn excluding_one_hop_removes_the_shared_corner_false_positive() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let features = tessellation.features();
    let separation = features.separation(&tessellation, Quantity::new(0.5), 1);
    assert!(
        separation.iter().all(|entry| entry.is_empty()),
        "{separation:?}"
    );
}

#[test]
fn triangles_that_share_no_nodes_have_no_features() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let triangles = match &tessellation.mesh().connectivities()[0] {
        Connectivity::Triangular(triangles) => triangles.iter().copied().collect::<Vec<_>>(),
        _ => panic!(),
    };
    let mut coordinates = Coordinates::zero(0);
    let mut unshared = Vec::new();
    triangles.iter().for_each(|&[a, b, c]| {
        let first = coordinates.len();
        [a, b, c]
            .into_iter()
            .for_each(|node| coordinates.push(tessellation.mesh().coordinates()[node].clone()));
        unshared.push([first, first + 1, first + 2])
    });
    let soup = Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(unshared.into())],
        coordinates,
    )));
    assert!(soup.features().corners().is_empty());
    assert!(soup.features().creases().is_empty());
}
