use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, tessellation::Tessellation},
    },
    math::{Tensor, TensorVec},
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
        (0..3).for_each(|d| assert!(corner[d] == 0.0 || corner[d] == 1.0, "{corner}"))
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
    let index = tessellation.features().index(0.2);
    let point = Coordinate::const_from([0.1, 0.1, 0.1]);
    let (corner, distance) = index.nearest_corner(&point, 0.2).unwrap();
    assert_eq!(index.corner(corner), &Coordinate::const_from([0.0; 3]));
    assert!((distance - 3.0_f64.sqrt() * 0.1).abs() < 1.0e-12);
    assert!(index.nearest_corner(&point, 0.1).is_none());
}

#[test]
fn a_crease_is_found_at_its_closest_point() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(0.3);
    let point = Coordinate::const_from([0.05, 0.05, 0.5]);
    let closest = index.nearest_crease(&point, 0.3).unwrap();
    assert!(
        (&closest - &Coordinate::const_from([0.0, 0.0, 0.5])).norm() < 1.0e-12,
        "{closest}"
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

#[test]
fn an_octahedron_is_all_creases_but_a_smoother_sphere_has_none() {
    let octahedron = sphere(0);
    assert_eq!(octahedron.features().creases().len(), 12);
    assert_eq!(octahedron.features().corners().len(), 6);
    let rounder = sphere(1);
    assert!(rounder.features().creases().is_empty());
    assert!(rounder.features().corners().is_empty());
}

fn quad(points: [[f64; 3]; 4]) -> [Coordinate<3>; 4] {
    points.map(Coordinate::const_from)
}

#[test]
fn a_crease_through_a_face_is_found_where_it_crosses() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(0.5);
    let face = quad([
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]);
    let through = index.through([&face[0], &face[1], &face[2], &face[3]]);
    assert_eq!(through.len(), 1);
    assert!(
        (&through[0].1 - &Coordinate::const_from([0.0, 0.0, 0.5])).norm() < 1.0e-12,
        "{}",
        through[0].1
    );
}

#[test]
fn a_face_clear_of_every_crease_finds_none() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(0.5);
    let face = quad([
        [0.2, 0.2, 0.5],
        [0.8, 0.2, 0.5],
        [0.8, 0.8, 0.5],
        [0.2, 0.8, 0.5],
    ]);
    assert!(
        index
            .through([&face[0], &face[1], &face[2], &face[3]])
            .is_empty()
    );
}

#[test]
fn a_crease_running_along_a_face_is_not_through_it() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(0.5);
    let face = quad([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]);
    let through = index.through([&face[0], &face[1], &face[2], &face[3]]);
    assert_eq!(through.len(), 1);
    assert!(
        (&through[0].1 - &Coordinate::const_from([0.0; 3])).norm() < 1.0e-12,
        "{}",
        through[0].1
    )
}

#[test]
fn both_cells_sharing_a_face_find_the_same_crossings() {
    let tessellation = cube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let index = tessellation.features().index(0.5);
    let face = quad([
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]);
    let one = index.through([&face[0], &face[1], &face[2], &face[3]]);
    let other = index.through([&face[2], &face[3], &face[0], &face[1]]);
    assert_eq!(one.len(), other.len());
    one.iter().zip(other.iter()).for_each(|(a, b)| {
        assert_eq!(a.0, b.0);
        assert!((&a.1 - &b.1).norm() < 1.0e-12, "{} {}", a.1, b.1)
    })
}
