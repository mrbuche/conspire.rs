use super::super::test::{box_surface, dual, hexahedron, sphere};
use super::super::{Class, Sign, Vertex};
use crate::{
    geometry::mesh::Connectivity,
    math::{CrossProduct, Tensor},
};
use std::collections::HashSet;

#[test]
fn tables_single_hexahedron() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
        .unwrap();
    assert_eq!(tables.signs().len(), 8);
    assert_eq!(
        tables
            .signs()
            .values()
            .filter(|&&sign| sign == Sign::Inside)
            .count(),
        4
    );
    assert_eq!(tables.crossings().len(), 4);
    tables.crossings().values().for_each(|points| {
        assert_eq!(points.len(), 1);
        points
            .iter()
            .for_each(|point| assert!((point.norm().value() - 1.0).abs() < 0.01))
    });
    assert_eq!(tables.segments().len(), 4);
    tables
        .segments()
        .values()
        .for_each(|segments| assert_eq!(segments.len(), 1))
}

#[test]
fn tables_generic_matches_hex_tables() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
        .unwrap();
    let generic = tessellation
        .tables_generic(&mesh, &classes, &HashSet::new())
        .unwrap();
    assert_eq!(generic.signs(), tables.signs());
    assert_eq!(generic.crossings(), tables.crossings());
    assert_eq!(generic.faces().len(), tables.faces().len());
    tables.faces().iter().for_each(|(key, corners)| {
        assert_eq!(generic.faces()[&key.to_vec()], corners.to_vec());
    });
    assert_eq!(generic.segments().len(), tables.segments().len());
    tables.segments().iter().for_each(|(key, pairs)| {
        assert_eq!(&generic.segments()[&key.to_vec()], pairs);
    });
}

#[test]
fn tables_sphere_dual() {
    let tessellation = sphere(3);
    let mesh = dual(&tessellation, 8.0);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
        .unwrap();
    assert!(!tables.crossings().is_empty());
    tables.crossings().values().flatten().for_each(|point| {
        let norm = point.norm().value();
        assert!((0.985..=1.0 + 1e-9).contains(&norm), "{norm}")
    });
    let coordinates = mesh.coordinates();
    tables.signs().iter().for_each(|(&node, &sign)| {
        let norm = coordinates[node].norm().value();
        if (norm - 1.0).abs() > 0.02 {
            assert_eq!(
                sign,
                if norm < 1.0 {
                    Sign::Inside
                } else {
                    Sign::Outside
                }
            )
        }
    });
    tables.segments().values().flatten().for_each(|segment| {
        segment.iter().for_each(|vertex| match vertex {
            Vertex::Node(_) => (),
            Vertex::Crossing(edge, ordinal) => {
                assert!(tables.crossings()[edge].len() > *ordinal)
            }
            Vertex::Feature(face, crease) => {
                assert!(tables.features().contains_key(&(*face, *crease)))
            }
        })
    })
}

#[test]
fn tables_double_crossing_edge() {
    let plate = box_surface([-2.0, -2.0, -0.05], [2.0, 2.0, 0.05]);
    let mesh = hexahedron([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let classes = plate.classify(&mesh);
    assert_eq!(classes, vec![Class::Cut]);
    let tables = plate.tables(&mesh, &classes, &HashSet::new()).unwrap();
    assert_eq!(
        tables
            .signs()
            .values()
            .filter(|&&sign| sign == Sign::Outside)
            .count(),
        8
    );
    [[0, 4], [1, 5], [2, 6], [3, 7]]
        .into_iter()
        .for_each(|[bottom, top]| {
            let points = &tables.crossings()[&[bottom, top]];
            assert_eq!(points.len(), 2);
            assert!(points[0][2] < points[1][2], "{points:?}");
            points.iter().for_each(|point| {
                assert!((point[2].abs().value() - 0.05).abs() < 1e-9, "{point:?}")
            });
        });
    let result = plate.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 1);
    match &result.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => {
            assert_eq!(hexes.iter().count(), 1);
            let element: Vec<usize> = hexes.iter().flatten().copied().collect();
            let coordinates = result.coordinates();
            coordinates.iter().for_each(|point| {
                assert!((point[2].abs().value() - 0.05).abs() < 1e-9, "{point:?}")
            });
            let volume = &(&coordinates[element[1]] - &coordinates[element[0]])
                .cross(&(&coordinates[element[3]] - &coordinates[element[0]]))
                * &(&coordinates[element[4]] - &coordinates[element[0]]);
            assert!((volume.value() - 0.4).abs() < 1e-9, "{volume}")
        }
        _ => panic!(),
    }
}
