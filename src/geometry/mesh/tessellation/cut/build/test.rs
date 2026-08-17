use super::super::geometry::signed_volume;
use super::super::test::{box_surface, hexahedron, signed_volumes, sphere};
use crate::{
    geometry::{
        mesh::{Connectivity, Mesh, tessellation::Tessellation},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::CrossProduct,
};
use std::collections::{HashMap, HashSet};

#[test]
fn assemble_generic_matches_assemble_hexahedron() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
        .unwrap();
    let expected = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    let expected_volume = match &expected.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => {
            let element: Vec<usize> = hexes.iter().flatten().copied().collect();
            let faces: Vec<Vec<usize>> = [
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [0, 3, 2, 1],
                [4, 5, 6, 7],
            ]
            .iter()
            .map(|face| face.iter().map(|&local| element[local]).collect())
            .collect();
            signed_volume(&faces, expected.coordinates())
        }
        _ => panic!("expected the recomposed hexahedron"),
    };
    let generic_tables = tessellation
        .tables_generic(&mesh, &classes, &HashSet::new())
        .unwrap();
    let result = tessellation
        .assemble_generic(&mesh, &classes, &generic_tables)
        .unwrap();
    assert_eq!(result.number_of_element_blocks(), 1);
    assert_eq!(result.number_of_nodes(), expected.number_of_nodes());
    match &result.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            assert_eq!(connectivity.elements_faces().len(), 1);
            assert_eq!(connectivity.faces_nodes().len(), 6);
            connectivity
                .faces_nodes()
                .iter()
                .for_each(|face| assert_eq!(face.len(), 4));
            let faces: Vec<Vec<usize>> = connectivity.elements_faces()[0]
                .iter()
                .map(|&face| connectivity.faces_nodes()[face].clone())
                .collect();
            let volume = signed_volume(&faces, result.coordinates());
            assert!(
                (volume - expected_volume).abs().value() < 1e-9,
                "{volume} vs {expected_volume}"
            );
        }
        _ => panic!("expected a polyhedral cell"),
    }
}

#[test]
fn assemble_generic_on_octree_polyhedron() {
    let tessellation = sphere(3);
    let mut octree =
        Octree::<u16, usize>::from_features(&tessellation, 4.0, CurvatureSizing::default(), 2);
    octree
        .equilibrate(Balancing::Weak(2), Pairing::Regular)
        .unwrap();
    let mesh: Mesh<3> = octree.into();
    match &mesh.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            signed_volumes(connectivity, mesh.coordinates())
                .iter()
                .for_each(|&volume| assert!(volume > 0.0, "raw octree mesh: {volume}"));
        }
        _ => panic!(),
    }
    assert!(matches!(
        &mesh.connectivities()[0],
        Connectivity::Polyhedral(connectivity) if connectivity.faces_nodes().iter().any(|face| face.len() > 4)
    ));
    let classes = tessellation.classify(&mesh);
    assert!(super::super::geometry::contained(&mesh, &classes));
    assert!(classes.contains(&super::super::Class::Cut));
    assert!(classes.contains(&super::super::Class::Inside));
    assert!(classes.contains(&super::super::Class::Outside));
    let (mesh, snapped) = tessellation.snap_generic(mesh, &classes).unwrap();
    let tables = tessellation
        .tables_generic(&mesh, &classes, &snapped)
        .unwrap();
    let result = tessellation
        .assemble_generic(&mesh, &classes, &tables)
        .unwrap();
    match &result.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            assert!(!connectivity.elements_faces().is_empty());
            connectivity
                .faces_nodes()
                .iter()
                .for_each(|face| assert!(face.len() > 2));
            connectivity
                .elements_faces()
                .iter()
                .for_each(|faces| assert!(faces.len() > 3));
            signed_volumes(connectivity, result.coordinates())
                .iter()
                .for_each(|&volume| assert!(volume > 0.0, "{volume}"));
        }
        _ => panic!("expected a polyhedral mesh"),
    }
}

fn tessellation_volume(tessellation: &Tessellation) -> f64 {
    let surface = tessellation.mesh();
    surface
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| {
            let coordinates = surface.coordinates();
            (coordinates[triangle[0]].cross(&coordinates[triangle[1]]) * &coordinates[triangle[2]])
                .value()
                / 6.0
        })
        .sum()
}

fn generic_cut(tessellation: &Tessellation, scale: f64) -> f64 {
    let mut octree =
        Octree::<u16, usize>::from_features(tessellation, scale, CurvatureSizing::default(), 2);
    octree
        .equilibrate(Balancing::Weak(2), Pairing::Regular)
        .unwrap();
    let mesh: Mesh<3> = octree.into();
    assert!(
        matches!(&mesh.connectivities()[0], Connectivity::Polyhedral(c)
            if c.faces_nodes().iter().any(|face| face.len() > 4)),
        "fixture no longer produces n-gon faces"
    );
    let classes = tessellation.classify(&mesh);
    assert!(super::super::geometry::contained(&mesh, &classes));
    let (mesh, snapped) = tessellation.snap_generic(mesh, &classes).unwrap();
    assert!(!snapped.is_empty(), "fixture no longer exercises snapping");
    let tables = tessellation
        .tables_generic(&mesh, &classes, &snapped)
        .unwrap();
    let result = tessellation
        .assemble_generic(&mesh, &classes, &tables)
        .unwrap();
    match &result.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            let mut uses = HashMap::new();
            connectivity.elements_faces().iter().for_each(|faces| {
                assert!(faces.len() > 3, "cell cannot bound a volume");
                faces
                    .iter()
                    .for_each(|&face| *uses.entry(face).or_insert(0) += 1)
            });
            uses.values()
                .for_each(|&count| assert!(count <= 2, "face shared by {count} cells"));
            connectivity
                .faces_nodes()
                .iter()
                .for_each(|face| assert!(face.len() > 2, "degenerate face"));
            let volumes = signed_volumes(connectivity, result.coordinates());
            volumes
                .iter()
                .for_each(|&volume| assert!(volume > 0.0, "inverted cell: {volume}"));
            volumes.iter().sum()
        }
        _ => panic!("expected a polyhedral mesh"),
    }
}

#[test]
fn generic_cut_octahedron() {
    let tessellation = sphere(1);
    let exact = tessellation_volume(&tessellation);
    let volume = generic_cut(&tessellation, 4.0);
    assert!(
        (volume - exact).abs() / exact < 1.0e-9,
        "{volume} vs {exact}"
    );
}

#[test]
fn generic_cut_box() {
    let tessellation = box_surface([-0.7, -0.55, -0.42], [0.63, 0.48, 0.71]);
    let exact = tessellation_volume(&tessellation);
    let volume = generic_cut(&tessellation, 4.0);
    assert!((volume - exact).abs() / exact < 0.1, "{volume} vs {exact}");
}
