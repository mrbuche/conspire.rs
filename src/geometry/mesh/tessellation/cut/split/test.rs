use super::super::test::{hexahedron, sphere};
use super::super::{
    Sign,
    face::{clip_face, face_cut},
    topology::{element_edges, element_faces},
};
use super::{Split, split_cell};
use crate::math::Tensor;
use std::collections::{HashMap, HashSet};

#[test]
fn split_cell_from_generic_tables() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let (mesh, snapped) = tessellation.snap(mesh, &classes).unwrap();
    let tables = tessellation
        .tables_generic(&mesh, &classes, &snapped)
        .unwrap();
    let block = mesh.iter().next().unwrap();
    let element = block.iter().next().unwrap();
    let faces = element_faces(block, element);
    let edges = element_edges(&faces);
    let mut crossing_ids = HashMap::<[usize; 2], Vec<usize>>::new();
    let mut next_id = mesh.coordinates().len();
    let mut crossed_edges: Vec<&[usize; 2]> = tables.crossings().keys().collect();
    crossed_edges.sort_unstable();
    crossed_edges.into_iter().for_each(|edge| {
        let ids: Vec<usize> = tables.crossings()[edge]
            .iter()
            .map(|_| {
                let id = next_id;
                next_id += 1;
                id
            })
            .collect();
        crossing_ids.insert(*edge, ids);
    });
    let mut face_cuts = HashMap::new();
    let mut face_polygons = HashMap::new();
    tables.faces().iter().for_each(|(key, corners)| {
        let cut = face_cut(corners, tables.signs(), tables.crossings()).unwrap();
        let polygons = if cut.flush {
            Vec::new()
        } else {
            clip_face(
                &cut,
                tables.segments().get(key),
                None,
                &crossing_ids,
                &HashMap::new(),
            )
        };
        face_polygons.insert(key.clone(), polygons);
        face_cuts.insert(key.clone(), cut);
    });
    let result = split_cell(
        &faces,
        &edges,
        tables.signs(),
        &face_cuts,
        tables.faces(),
        &face_polygons,
        tables.segments(),
        &crossing_ids,
    )
    .unwrap();
    let Split::Cut(cut_cell) = result else {
        panic!("expected the hexahedron to be cut")
    };
    assert!(!cut_cell.polygons.is_empty());
    assert!(cut_cell.clipped <= cut_cell.polygons.len());
    let outside_nodes: HashSet<usize> = tables
        .signs()
        .iter()
        .filter(|&(_, &sign)| sign == Sign::Outside)
        .map(|(&node, _)| node)
        .collect();
    cut_cell.polygons.iter().for_each(|polygon| {
        assert!(polygon.len() > 2);
        polygon
            .iter()
            .for_each(|node| assert!(!outside_nodes.contains(node)))
    });
}
