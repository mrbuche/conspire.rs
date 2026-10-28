use super::{Defects, Watertight};
use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Connectivity, Mesh, Tessellation,
            test::{CONNECTIVITY, COORDINATES, cuboid, sphere},
        },
        solid::SolidOracle,
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Volume,
};

fn tessellation(faces: Vec<[usize; 3]>, coordinates: Vec<Coordinate<3>>) -> Tessellation {
    Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        coordinates.into(),
    ))
    .into()
}

fn cube(faces: Vec<[usize; 3]>) -> Tessellation {
    tessellation(faces, COORDINATES.to_vec())
}

fn unit() -> Vec<[usize; 3]> {
    CONNECTIVITY.to_vec()
}

fn flipped(face: [usize; 3]) -> [usize; 3] {
    [face[0], face[2], face[1]]
}

#[test]
fn a_cube_has_no_defects() -> Result<(), AssertionError> {
    let defects = cube(unit()).defects();
    assert!(defects.is_empty(), "{defects:?}");
    Assert::default().eq_within_tols(
        cube(unit()).enclosed_volume(),
        &Quantity::<Volume>::new(1.0),
    )
}

#[test]
fn a_missing_face_leaves_boundary_edges() {
    let mut faces = unit();
    faces.pop();
    let defects = cube(faces).defects();
    assert_eq!(defects.boundary_edges.len(), 3);
    assert!(defects.non_manifold_edges.is_empty());
    assert!(defects.inconsistent_edges.is_empty());
    assert!(!defects.is_empty());
}

#[test]
fn a_flipped_face_is_inconsistent_but_not_open() {
    let mut faces = unit();
    faces[4] = flipped(faces[4]);
    let defects = cube(faces).defects();
    assert_eq!(defects.inconsistent_edges.len(), 3);
    assert!(defects.boundary_edges.is_empty());
}

#[test]
fn a_cube_turned_inside_out_is_only_inward() {
    let faces = unit().into_iter().map(flipped).collect();
    let defects = cube(faces).defects();
    assert!(defects.inward);
    assert!(defects.boundary_edges.is_empty());
    assert!(defects.non_manifold_edges.is_empty());
    assert!(defects.non_manifold_vertices.is_empty());
    assert!(defects.inconsistent_edges.is_empty());
    assert!(defects.self_intersections.is_empty());
}

#[test]
fn a_third_face_on_an_edge_is_non_manifold() {
    let mut faces = unit();
    let mut coordinates = COORDINATES.to_vec();
    coordinates.push([0.5, -1.0, -1.0].into());
    faces.push([0, 1, 8]);
    let defects = tessellation(faces, coordinates).defects();
    let edges: Vec<[usize; 2]> = defects
        .non_manifold_edges
        .iter()
        .map(|&[a, b]| [a.min(b), a.max(b)])
        .collect();
    assert_eq!(edges, vec![[0, 1]]);
}

#[test]
fn cubes_meeting_at_a_vertex_pinch() {
    let mut faces = unit();
    let mut coordinates = COORDINATES.to_vec();
    let other = cuboid([1.0; 3], [2.0; 3]);
    let remap: Vec<usize> = (0..8)
        .map(|node| {
            if node == 0 {
                6
            } else {
                coordinates.push(other.mesh().coordinates()[node].clone());
                coordinates.len() - 1
            }
        })
        .collect();
    faces.extend(CONNECTIVITY.iter().map(|face| face.map(|node| remap[node])));
    let defects = tessellation(faces, coordinates).defects();
    assert_eq!(defects.non_manifold_vertices, vec![6]);
    assert!(defects.boundary_edges.is_empty());
}

#[test]
fn overlapping_cubes_in_one_tessellation_intersect_themselves() {
    let mut faces = unit();
    let mut coordinates = COORDINATES.to_vec();
    let other = cuboid([0.5; 3], [1.5; 3]);
    coordinates.extend(other.mesh().coordinates().iter().cloned());
    faces.extend(CONNECTIVITY.iter().map(|face| face.map(|node| node + 8)));
    let defects = tessellation(faces, coordinates).defects();
    assert!(!defects.self_intersections.is_empty());
    assert!(defects.boundary_edges.is_empty());
    assert!(!defects.inward);
}

#[test]
fn watertight_accepts_closed_surfaces() {
    assert!(Watertight::new(cuboid([0.0; 3], [1.0; 3])).is_ok());
    assert!(Watertight::new(sphere(8, 16, 1.0)).is_ok());
}

#[test]
fn watertight_rejects_a_flipped_face_and_an_inside_out_surface() {
    let mut faces = unit();
    faces[4] = flipped(faces[4]);
    assert!(Watertight::new(cube(faces)).is_err());
    assert!(Watertight::new(cube(unit().into_iter().map(flipped).collect())).is_err());
}

#[test]
fn watertight_returns_the_defects_of_an_open_surface() {
    let mut faces = unit();
    faces.pop();
    let Err(defects): Result<Watertight, Defects> = Watertight::new(cube(faces)) else {
        panic!("an open surface was accepted")
    };
    assert_eq!(defects.boundary_edges.len(), 3);
}

#[test]
fn watertight_answers_as_its_tessellation_does() {
    let watertight = Watertight::new(cuboid([0.0; 3], [1.0; 3])).unwrap();
    let point = [2.0, 0.5, 0.5].into();
    assert_eq!(
        watertight.signed_distance(&point),
        watertight.tessellation().signed_distance(&point)
    );
    assert!(watertight.project(&point).unwrap().exact);
}
