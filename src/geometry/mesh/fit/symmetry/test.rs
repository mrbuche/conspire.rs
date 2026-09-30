use super::Symmetry;
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Dualization, Mesh, Tessellation, test::octahedron},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};

fn background(target: &Tessellation, scale: Scalar) -> Mesh<3> {
    let mut octree = Octree::<u16, usize>::from_features(
        target,
        scale,
        CurvatureSizing {
            tolerance: None,
            ..Default::default()
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let mut mesh = octree.dualize();
    target.trim(&mut mesh).unwrap();
    mesh
}

fn moved(mesh: &Mesh<3>, vertex: usize, to: [Scalar; 3]) -> Mesh<3> {
    let mut coordinates = mesh.coordinates().clone();
    coordinates[vertex] = Coordinate::const_from(to);
    let triangles: Vec<[usize; 3]> = mesh
        .iter()
        .flatten()
        .map(|element| [element[0], element[1], element[2]])
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    ))
}

#[test]
fn detects_the_three_mirror_planes() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let symmetry = Symmetry::detect(&mesh, &target).unwrap();
    assert_eq!(symmetry.signs.len(), 8);
    assert_eq!(symmetry.images.len(), 8);
    symmetry
        .center
        .iter()
        .for_each(|entry| assert!(entry.abs() < 1.0e-12, "center: {:?}", symmetry.center));
    symmetry
        .images
        .iter()
        .for_each(|images| assert_eq!(images.len(), mesh.coordinates().len()));
}

#[test]
fn rejects_a_mesh_with_one_node_displaced() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let (connectivities, mut coordinates) = mesh.into();
    let point = coordinates[0].clone();
    coordinates[0] =
        Coordinate::const_from([point[0].value() + 0.01, point[1].value(), point[2].value()]);
    let mesh = Mesh::from((connectivities.into_members(), coordinates));
    assert!(Symmetry::detect(&mesh, &target).is_none());
}

#[test]
fn rejects_a_target_respecting_no_mirror_plane() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let skewed = Tessellation::from(moved(target.mesh(), 0, [1.2, 0.05, 0.03]));
    assert!(Symmetry::detect(&mesh, &skewed).is_none());
}

#[test]
fn keeps_only_the_mirror_planes_the_target_respects() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let skewed = Tessellation::from(moved(target.mesh(), 0, [1.2, 0.05, 0.0]));
    let symmetry = Symmetry::detect(&mesh, &skewed).unwrap();
    assert_eq!(symmetry.signs, vec![[1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]);
}

#[test]
fn orbit_averages_are_symmetric_and_idempotent() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let symmetry = Symmetry::detect(&mesh, &target).unwrap();
    let nodes: Vec<usize> = (0..mesh.coordinates().len()).collect();
    let orbits = symmetry.orbits(&nodes).unwrap();
    let field: Vec<[Scalar; 3]> = mesh
        .coordinates()
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let phase = index as Scalar;
            [
                point[0].value() + 0.01 * phase.sin(),
                point[1].value() + 0.01 * (2.0 * phase).cos(),
                point[2].value() + 0.01 * (3.0 * phase).sin(),
            ]
        })
        .collect();
    let projected = orbits.points(&field);
    let again = orbits.points(&projected);
    projected.iter().zip(&again).for_each(|(a, b)| {
        (0..3).for_each(|i| assert!((a[i] - b[i]).abs() < 1.0e-12));
    });
    orbits
        .images
        .iter()
        .zip(&orbits.signs)
        .for_each(|(images, signs)| {
            projected.iter().enumerate().for_each(|(position, point)| {
                let image = &projected[images[position]];
                (0..3).for_each(|i| {
                    let expected = orbits.center[i] + signs[i] * (point[i] - orbits.center[i]);
                    assert!((image[i] - expected).abs() < 1.0e-12);
                });
            });
        });
    let vectors = orbits.vectors(&field);
    orbits
        .images
        .iter()
        .zip(&orbits.signs)
        .for_each(|(images, signs)| {
            vectors.iter().enumerate().for_each(|(position, vector)| {
                let image = &vectors[images[position]];
                (0..3).for_each(|i| assert!((image[i] - signs[i] * vector[i]).abs() < 1.0e-12));
            });
        });
}

#[test]
fn a_subset_of_nodes_must_be_closed_under_the_group() {
    let target = octahedron(2);
    let mesh = background(&target, 5.0);
    let symmetry = Symmetry::detect(&mesh, &target).unwrap();
    assert!(symmetry.orbits(&[0, 1, 2]).is_none());
}
