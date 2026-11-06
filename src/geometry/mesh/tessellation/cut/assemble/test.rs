use super::super::test::{hexahedron, signed_volumes, sphere};
use super::super::{Class, geometry::star_volume};
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Tessellation},
    },
    math::{CrossProduct, FxHashSet, Quantity},
};

#[test]
fn assemble_single_hexahedron() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &FxHashSet::default())
        .unwrap();
    let result = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 1);
    assert_eq!(result.number_of_nodes(), 8);
    match &result.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => {
            assert_eq!(hexes.iter().count(), 1);
            let element: Vec<usize> = hexes.iter().flatten().copied().collect();
            let coordinates = result.coordinates();
            let base = &(&coordinates[element[1]] - &coordinates[element[0]])
                .cross(&(&coordinates[element[3]] - &coordinates[element[0]]))
                * &(&coordinates[element[4]] - &coordinates[element[0]]);
            assert!(base > Quantity::default())
        }
        _ => panic!(),
    }
}

#[test]
fn agglomerate_sliver() {
    let coordinates = Coordinates::from(vec![
        [-1.0, -1.0, -2.0],
        [3.0, -1.0, -2.0],
        [3.0, 2.0, -2.0],
        [-1.0, 2.0, -2.0],
        [-1.0, -1.0, -0.04],
        [3.0, -1.0, 0.28],
        [3.0, 2.0, 0.28],
        [-1.0, 2.0, -0.04],
    ]);
    let triangles = vec![
        [0, 3, 2],
        [0, 2, 1],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [3, 7, 6],
        [3, 6, 2],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let tessellation = Tessellation::from((triangles, coordinates));
    let mesh = (
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]].into(),
        )],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
        .into(),
    )
        .into();
    let classes = tessellation.classify(&mesh);
    assert_eq!(classes, vec![Class::Cut, Class::Cut]);
    let tables = tessellation
        .tables(&mesh, &classes, &FxHashSet::default())
        .unwrap();
    let result = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 1);
    match &result.connectivities()[0] {
        Connectivity::Polyhedral(polyhedra) => {
            assert_eq!(polyhedra.elements_faces().len(), 1);
            assert_eq!(polyhedra.elements_faces()[0].len(), 9);
            let faces: Vec<Vec<usize>> = polyhedra.elements_faces()[0]
                .iter()
                .map(|&face| polyhedra.faces_nodes()[face].clone())
                .collect();
            let volume = star_volume(&faces, result.coordinates());
            assert!(
                (volume - Quantity::new(0.220095389507154)).abs() < Quantity::new(1e-12),
                "{volume}"
            );
            let signed = signed_volumes(polyhedra, result.coordinates())[0];
            assert!((signed - 0.220095389507154).abs() < 1e-12, "{signed}")
        }
        _ => panic!(),
    }
}
