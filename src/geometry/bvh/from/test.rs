use crate::geometry::{
    Coordinate, Coordinates,
    bvh::{BoundingVolumeHierarchy, ray::Ray},
    mesh::{Connectivity, Mesh, Tessellation},
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

fn downward_ray() -> Ray<3> {
    Ray::from((
        Coordinate::const_from([0.2, 0.2, 5.0]),
        Coordinate::const_from([0.0, 0.0, -1.0]),
    ))
}

#[test]
fn from_mesh() {
    let mesh = mesh();
    let bvh = BoundingVolumeHierarchy::from(&mesh);
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let hit = bvh
        .intersect(&downward_ray(), mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(hit.index(), 1);
    assert_eq!(hit.distance(), 3.0);
}

#[test]
fn from_tessellation_matches_mesh() {
    let tessellation = Tessellation::from(mesh());
    let bvh = BoundingVolumeHierarchy::from(&tessellation);
    let mesh = tessellation.mesh();
    let elements: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    let hit = bvh
        .intersect(&downward_ray(), mesh.coordinates(), &elements)
        .unwrap();
    assert_eq!(hit.index(), 1);
    assert_eq!(hit.distance(), 3.0);
}
