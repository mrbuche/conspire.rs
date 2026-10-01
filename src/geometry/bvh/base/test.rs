use super::closest_point_on_triangle;
use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            test::{octahedron, perpendicular_facet},
        },
    },
    math::{CrossProduct, Quantity, Tensor, assert::AssertionError},
};

const CONNECTIVITY: [[usize; 3]; 2] = [[0, 1, 2], [3, 4, 5]];

const COORDINATES: [Coordinate<3>; 6] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 2.0]),
    Coordinate::const_from([1.0, 0.0, 2.0]),
    Coordinate::const_from([0.0, 1.0, 2.0]),
];

fn mesh() -> Mesh<3> {
    let connectivities = vec![Connectivity::Triangular(CONNECTIVITY.to_vec().into())];
    let coordinates = Coordinates::from(COORDINATES);
    Mesh::from((connectivities, coordinates))
}

#[test]
fn hits_nearest_triangle() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let ray = (
        Coordinate::const_from([0.2, 0.2, 5.0]),
        Coordinate::const_from([0.0, 0.0, -1.0]),
    )
        .into();
    let hit = bvh.intersect(&ray, mesh.coordinates(), &elements).unwrap();
    assert_eq!(hit.index(), 1);
    assert!((hit.distance().value() - 3.0).abs() < 1e-12);
}

#[test]
fn misses_when_outside_triangle() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let ray = (
        Coordinate::const_from([0.9, 0.9, 5.0]),
        Coordinate::const_from([0.0, 0.0, -1.0]),
    )
        .into();
    assert_eq!(bvh.intersect(&ray, mesh.coordinates(), &elements), None);
}

#[test]
fn pointing_away_misses() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let ray = (
        Coordinate::const_from([0.2, 0.2, 5.0]),
        Coordinate::const_from([0.0, 0.0, 1.0]),
    )
        .into();
    assert_eq!(bvh.intersect(&ray, mesh.coordinates(), &elements), None);
}

#[test]
fn closest_point_projects_onto_nearest_face() -> Result<(), AssertionError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let query = Coordinate::const_from([0.2, 0.2, 0.5]);
    let (point, index) = bvh
        .closest_point(&query, mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(index, 0);
    Assert::default().eq_within_tols(&point, &Coordinate::const_from([0.2, 0.2, 0.0]))
}

#[test]
fn closest_point_clamps_to_vertex() -> Result<(), AssertionError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let query = Coordinate::const_from([-1.0, -1.0, 0.0]);
    let (point, index) = bvh
        .closest_point(&query, mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(index, 0);
    Assert::default().eq_within_tols(&point, &Coordinate::const_from([0.0, 0.0, 0.0]))
}

#[test]
fn intersect_excluding_finds_the_far_side_from_a_perpendicular_facet() {
    for size in [0.001, 0.01] {
        for axis in 0..3 {
            for sign in [1.0, -1.0] {
                let mesh = perpendicular_facet(axis, sign, size);
                let bvh = BoundingVolumeHierarchy::from(&mesh);
                let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
                let coordinates = mesh.coordinates();
                let centroid = (&coordinates[0] + &coordinates[1] + &coordinates[2]) / 3.0;
                let far = (&coordinates[3] + &coordinates[4] + &coordinates[5]) / 3.0;
                let normal = (&coordinates[1] - &coordinates[0])
                    .cross(&coordinates[2] - &coordinates[0])
                    .normalized();
                let toward = if (&normal * &(&far - &centroid)).value() > 0.0 {
                    normal
                } else {
                    -&normal
                };
                let ray = (centroid, toward).into();
                let hit = bvh
                    .intersect_excluding(&ray, coordinates, &elements, 0)
                    .unwrap();
                assert_eq!(hit.index(), 1, "size {size}, axis {axis}, sign {sign}");
                assert!(
                    (hit.distance().value() - 2.0).abs() < 0.01,
                    "size {size}, axis {axis}, sign {sign}: {}",
                    hit.distance().value()
                );
            }
        }
    }
}

#[test]
fn closest_points_returns_every_equidistant_face() -> Result<(), AssertionError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let query = Coordinate::const_from([0.2, 0.2, 1.0]);
    let ties = bvh.closest_points(&query, mesh.coordinates(), &elements, Quantity::new(1.0e-9));
    assert_eq!(ties.len(), 2);
    assert_eq!(ties[0].1, 0);
    assert_eq!(ties[1].1, 1);
    Assert::default().eq_within_tols(&ties[0].0, &Coordinate::const_from([0.2, 0.2, 0.0]))?;
    Assert::default().eq_within_tols(&ties[1].0, &Coordinate::const_from([0.2, 0.2, 2.0]))
}

#[test]
fn closest_points_excludes_faces_beyond_the_tolerance() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let query = Coordinate::const_from([0.2, 0.2, 0.5]);
    let ties = bvh.closest_points(&query, mesh.coordinates(), &elements, Quantity::new(1.0e-9));
    assert_eq!(ties.len(), 1);
    assert_eq!(ties[0].1, 0);
    let ties = bvh.closest_points(&query, mesh.coordinates(), &elements, Quantity::new(0.9));
    assert_eq!(ties.len(), 1);
    let ties = bvh.closest_points(&query, mesh.coordinates(), &elements, Quantity::new(1.1));
    assert_eq!(ties.len(), 2);
}

#[test]
fn closest_points_agrees_with_closest_point() -> Result<(), AssertionError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let query = Coordinate::const_from([-1.0, -1.0, 0.0]);
    let (point, index) = bvh
        .closest_point(&query, mesh.coordinates(), &elements)
        .unwrap();
    let ties = bvh.closest_points(&query, mesh.coordinates(), &elements, Quantity::new(1.0e-9));
    assert_eq!(ties.len(), 1);
    assert_eq!(ties[0].1, index);
    Assert::default().eq_within_tols(&ties[0].0, &point)
}

#[test]
fn closest_points_matches_a_scan_of_every_face() {
    let tessellation = octahedron(3);
    let surface = tessellation.mesh();
    let bvh = BoundingVolumeHierarchy::from(surface);
    let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let coordinates = surface.coordinates();
    let levels = [-0.9, -0.3, 0.0, 0.3, 0.9, 1.6];
    for &x in &levels {
        for &y in &levels {
            for &z in &levels {
                let query = Coordinate::const_from([x, y, z]);
                let scan: Vec<(Coordinate<3>, usize, f64)> = elements
                    .iter()
                    .enumerate()
                    .map(|(index, element)| {
                        let candidate = closest_point_on_triangle(
                            &query,
                            &coordinates[element[0]],
                            &coordinates[element[1]],
                            &coordinates[element[2]],
                        );
                        let distance = (&candidate - &query).norm().value();
                        (candidate, index, distance)
                    })
                    .collect();
                let nearest = scan
                    .iter()
                    .map(|&(_, _, d)| d)
                    .fold(f64::INFINITY, f64::min);
                for tolerance in [1.0e-9, 0.02, 0.4] {
                    let expected: Vec<&(Coordinate<3>, usize, f64)> = scan
                        .iter()
                        .filter(|&&(_, _, distance)| distance <= nearest + tolerance)
                        .collect();
                    let found = bvh.closest_points(
                        &query,
                        coordinates,
                        &elements,
                        Quantity::new(tolerance),
                    );
                    assert_eq!(
                        found.iter().map(|&(_, index)| index).collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|&&(_, index, _)| index)
                            .collect::<Vec<_>>(),
                        "query {x} {y} {z}, tolerance {tolerance}"
                    );
                    found
                        .iter()
                        .zip(&expected)
                        .for_each(|((point, _), expected)| {
                            assert!((point - &expected.0).norm().value() < 1.0e-12);
                        });
                }
            }
        }
    }
}

#[test]
fn intersect_excluding_matches_intersect_when_nothing_is_excluded() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let ray = (
        Coordinate::const_from([0.2, 0.2, 5.0]),
        Coordinate::const_from([0.0, 0.0, -1.0]),
    )
        .into();
    let nearest = bvh.intersect(&ray, mesh.coordinates(), &elements).unwrap();
    let skipped = bvh
        .intersect_excluding(&ray, mesh.coordinates(), &elements, 0)
        .unwrap();
    assert_eq!(nearest.index(), 1);
    assert_eq!(skipped.index(), 1);
    let past = bvh
        .intersect_excluding(&ray, mesh.coordinates(), &elements, 1)
        .unwrap();
    assert_eq!(past.index(), 0);
    assert!((past.distance().value() - 5.0).abs() < 1e-12);
}
