use super::super::geometry::signed_volume;
use super::super::test::{hexahedron, signed_volumes, sphere};
use super::assemble_generic;
use crate::{
    geometry::{
        mesh::{Connectivity, Mesh, Output, Vtk, tessellation::Tessellation},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    io::{Write, write::Compression},
};
use std::{collections::HashSet, path::Path};

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
    let result = assemble_generic(&mesh, &classes, &generic_tables).unwrap();
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
                (volume - expected_volume).abs() < 1e-9,
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
    assert!(classes.contains(&super::super::Class::Cut));
    assert!(classes.contains(&super::super::Class::Inside));
    assert!(classes.contains(&super::super::Class::Outside));
    let (mesh, snapped) = tessellation.snap_generic(mesh, &classes).unwrap();
    let tables = tessellation
        .tables_generic(&mesh, &classes, &snapped)
        .unwrap();
    let result = assemble_generic(&mesh, &classes, &tables).unwrap();
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

#[test]
#[ignore = "needs bone_tri.stl at the repo root; not a committed fixture"]
fn hex_cut_on_bone() {
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    [3.0, 4.0, 8.0].into_iter().for_each(|scale| {
        let mesh = tessellation.cut(Balancing::Strong, scale).unwrap();
        mesh.iter().for_each(|block| {
            if let Connectivity::Polyhedral(connectivity) = block {
                signed_volumes(connectivity, mesh.coordinates())
                    .iter()
                    .for_each(|&volume| assert!(volume > 0.0, "scale {scale}: {volume}"))
            }
        })
    })
}

#[test]
#[ignore = "needs bone_tri.stl at the repo root; not a committed fixture"]
fn assemble_generic_on_bone() {
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    let mut octree =
        Octree::<u16, usize>::from_features(&tessellation, 4.0, CurvatureSizing::default(), 2);
    octree
        .equilibrate(Balancing::Weak(2), Pairing::Regular)
        .unwrap();
    let mesh: Mesh<3> = octree.into();
    let classes = tessellation.classify(&mesh);
    let (mesh, snapped) = tessellation.snap_generic(mesh, &classes).unwrap();
    let tables = tessellation
        .tables_generic(&mesh, &classes, &snapped)
        .unwrap();
    let result = assemble_generic(&mesh, &classes, &tables).unwrap();
    match &result.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            assert!(!connectivity.elements_faces().is_empty());
            connectivity
                .elements_faces()
                .iter()
                .for_each(|faces| assert!(faces.len() > 3));
            signed_volumes(connectivity, result.coordinates())
                .iter()
                .for_each(|&volume| assert!(volume > 0.0, "{volume}"))
        }
        _ => panic!("expected a polyhedral mesh"),
    }
    result
        .write(Output::Vtk(Vtk::UnstructuredGrid(Compression::Off(
            "bone_cut_generic.vtu",
        ))))
        .unwrap()
}
