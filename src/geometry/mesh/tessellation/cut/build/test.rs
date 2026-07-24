use super::super::geometry::signed_volume;
use super::super::test::{hexahedron, sphere};
use super::assemble_generic;
use crate::geometry::mesh::Connectivity;
use std::collections::HashSet;

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
