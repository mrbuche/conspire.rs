use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{Connectivity, Mesh},
    },
    math::test::{TestError, assert_eq_within_tols},
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
    assert_eq!(hit.distance(), 3.0);
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
fn closest_point_projects_onto_nearest_face() -> Result<(), TestError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    // above triangle 0 (z = 0) and below triangle 1 (z = 2), inside the triangle laterally
    let query = Coordinate::const_from([0.2, 0.2, 0.5]);
    let (point, index) = bvh
        .closest_point(&query, mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(index, 0);
    assert_eq_within_tols(&point, &Coordinate::const_from([0.2, 0.2, 0.0]))
}

#[test]
fn closest_point_clamps_to_vertex() -> Result<(), TestError> {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    // beyond the corner at node 0: closest point is that vertex itself
    let query = Coordinate::const_from([-1.0, -1.0, 0.0]);
    let (point, index) = bvh
        .closest_point(&query, mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(index, 0);
    assert_eq_within_tols(&point, &Coordinate::const_from([0.0, 0.0, 0.0]))
}
