use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, Partition, partition::test::blocks},
    },
    math::CrossProduct,
};
use std::collections::HashMap;

fn signed_volumes(mesh: &Mesh<3>) -> Vec<f64> {
    let Some(Connectivity::Polyhedral(polyhedra)) = mesh.iter().next() else {
        panic!("expected polyhedra")
    };
    let coordinates = mesh.coordinates();
    let faces_nodes = polyhedra.faces_nodes();
    let mut owner = HashMap::new();
    polyhedra
        .elements_faces()
        .iter()
        .enumerate()
        .for_each(|(cell, faces)| {
            faces.iter().for_each(|&face| {
                owner.entry(face).or_insert(cell);
            })
        });
    polyhedra
        .elements_faces()
        .iter()
        .enumerate()
        .map(|(cell, faces)| {
            faces
                .iter()
                .map(|&face| {
                    let nodes = &faces_nodes[face];
                    let middle = nodes
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .sum::<Coordinate<3>>()
                        / nodes.len() as f64;
                    let volume: f64 = (0..nodes.len())
                        .map(|i| {
                            let one = &coordinates[nodes[i]];
                            let two = &coordinates[nodes[(i + 1) % nodes.len()]];
                            (&middle * &one.cross(two)).value() / 6.0
                        })
                        .sum();
                    if owner[&face] == cell {
                        volume
                    } else {
                        -volume
                    }
                })
                .sum()
        })
        .collect()
}

fn assert_volumes(mesh: &Mesh<3>, expected: &[f64]) {
    let volumes = signed_volumes(mesh);
    assert_eq!(volumes.len(), expected.len());
    volumes
        .iter()
        .zip(expected)
        .for_each(|(volume, expected)| assert!((volume - expected).abs() < 1e-12, "{volume}"));
}

fn number_of_faces(mesh: &Mesh<3>) -> usize {
    mesh.number_of_faces().unwrap()
}

#[test]
fn hexes_box_split_has_shared_interface_and_no_interior_nodes() {
    let mesh = blocks([4, 2, 2]);
    let agglomerated = mesh.partition_box([2, 1, 1]).agglomerate(&mesh).unwrap();
    assert_eq!(agglomerated.number_of_elements(), 2);
    assert_eq!(number_of_faces(&agglomerated), 24 + 24 - 4);
    assert_eq!(agglomerated.number_of_nodes(), 45 - 2);
    assert_volumes(&agglomerated, &[8.0, 8.0]);
}

#[test]
fn hexes_volumes_are_conserved_for_many_parts() {
    let mesh = blocks([6, 6, 6]);
    let agglomerated = mesh.partition_box([2, 2, 2]).agglomerate(&mesh).unwrap();
    assert_volumes(&agglomerated, &[27.0; 8]);
    let agglomerated = mesh.partition_rcb(5).agglomerate(&mesh).unwrap();
    let volumes = signed_volumes(&agglomerated);
    assert!(volumes.iter().all(|&volume| volume > 0.0));
    assert!((volumes.iter().sum::<f64>() - 216.0).abs() < 1e-9);
}

#[test]
fn tets_are_agglomerated() {
    let mesh = Mesh::from_lattice_tets(
        (0..4).map(|i| ([i, 0, 0], 1)),
        [4, 1, 1],
        &Coordinate::from([1.0, 1.0, 1.0]),
        &Coordinate::from([0.0, 0.0, 0.0]),
    );
    let agglomerated = mesh.partition_box([2, 1, 1]).agglomerate(&mesh).unwrap();
    assert_volumes(&agglomerated, &[2.0, 2.0]);
}

#[test]
fn mixed_elements_are_agglomerated() {
    let coordinates = Coordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.5, 0.5],
    ]);
    let mesh = Mesh::from((
        vec![
            Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
            Connectivity::Tetrahedral(vec![[1, 2, 6, 8], [1, 6, 5, 8]].into()),
        ],
        coordinates,
    ));
    let agglomerated = Partition::new(&mesh, vec![0, 1, 1])
        .agglomerate(&mesh)
        .unwrap();
    let volumes = signed_volumes(&agglomerated);
    assert!((volumes[0] - 1.0).abs() < 1e-12, "{volumes:?}");
    assert!((volumes[1] - 1.0 / 3.0).abs() < 1e-12, "{volumes:?}");
}

#[test]
fn polyhedra_are_agglomerated_again() {
    let mesh = blocks([6, 1, 1]);
    let once = mesh.partition_box([3, 1, 1]).agglomerate(&mesh).unwrap();
    assert_volumes(&once, &[2.0, 2.0, 2.0]);
    let twice = Partition::new(&once, vec![0, 0, 1])
        .agglomerate(&once)
        .unwrap();
    assert_volumes(&twice, &[4.0, 2.0]);
    let thrice = Partition::new(&twice, vec![0, 0])
        .agglomerate(&twice)
        .unwrap();
    assert_volumes(&thrice, &[6.0]);
}

#[test]
fn disconnected_part_is_an_error() {
    let mesh = blocks([4, 1, 1]);
    assert!(
        Partition::new(&mesh, vec![0, 1, 0, 1])
            .agglomerate(&mesh)
            .is_err()
    );
}

#[test]
fn parts_touching_only_at_an_edge_are_an_error() {
    let mesh = blocks([2, 2, 1]);
    assert!(
        Partition::new(&mesh, vec![0, 1, 1, 0])
            .agglomerate(&mesh)
            .is_err()
    );
}

#[test]
fn surface_elements_are_an_error() {
    let mesh = crate::geometry::mesh::test::mesh();
    assert!(
        Partition::new(&mesh, vec![0; mesh.number_of_elements()])
            .agglomerate(&mesh)
            .is_err()
    );
}

#[test]
fn polyhedra_owned_by_a_later_part_are_reoriented() {
    let mesh = blocks([6, 1, 1]);
    let once = mesh.partition_box([3, 1, 1]).agglomerate(&mesh).unwrap();
    let twice = Partition::new(&once, vec![1, 1, 0])
        .agglomerate(&once)
        .unwrap();
    assert_volumes(&twice, &[2.0, 4.0]);
}
