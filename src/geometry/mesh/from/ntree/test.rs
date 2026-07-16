use crate::{
    geometry::{
        mesh::{Connectivity, Mesh},
        ntree::{
            Balancing, Octree, Pairing, Quadtree, Rescaling,
            node::{Kind, Node},
        },
    },
    math::Tensor,
};
use std::collections::HashSet;

fn octree(length: u16) -> Octree<u16, usize> {
    Octree {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0; 3],
            length,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [length as f64 / 2.0; 3],
            cell: 1.0,
            half: length as f64 / 2.0,
        },
    }
}

fn quadtree(length: u16) -> Quadtree<u16, usize> {
    Quadtree {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0; 2],
            length,
            facets: [None; 4],
            kind: Kind::Leaf,
            value: None,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [length as f64 / 2.0; 2],
            cell: 1.0,
            half: length as f64 / 2.0,
        },
    }
}

fn polytopal<const D: usize>(mesh: &Mesh<D>) -> (&[Vec<usize>], &[Vec<usize>]) {
    match &mesh.connectivities()[0] {
        Connectivity::Polyhedral(connectivity) => {
            (connectivity.elements_faces(), connectivity.faces_nodes())
        }
        Connectivity::Polygonal(connectivity) => {
            (connectivity.elements_faces(), connectivity.faces_nodes())
        }
        _ => panic!("expected polytopal connectivity"),
    }
}

fn point<const D: usize>(mesh: &Mesh<D>, node: usize) -> [f64; D] {
    std::array::from_fn(|axis| mesh.coordinates()[node][axis])
}

fn conformal<const D: usize>(mesh: &Mesh<D>) {
    let (elements_faces, faces_nodes) = polytopal(mesh);
    let mut references = vec![0; faces_nodes.len()];
    elements_faces
        .iter()
        .flatten()
        .for_each(|&face| references[face] += 1);
    references
        .iter()
        .enumerate()
        .for_each(|(face, &count)| assert!(count == 1 || count == 2, "face {face}: {count} refs"));
    let mut sets = HashSet::new();
    faces_nodes.iter().for_each(|face| {
        let mut sorted = face.clone();
        sorted.sort_unstable();
        assert!(sets.insert(sorted), "duplicate face {face:?}");
    });
}

fn volume(mesh: &Mesh<3>) -> f64 {
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let (elements_faces, faces_nodes) = polytopal(mesh);
    let mut total = 0.0;
    elements_faces.iter().for_each(|element_faces| {
        let nodes = element_faces
            .iter()
            .flat_map(|&face| faces_nodes[face].iter().copied())
            .collect::<HashSet<_>>();
        let mut centroid = [0.0; 3];
        nodes.iter().for_each(|&node| {
            let x = point(mesh, node);
            (0..3).for_each(|axis| centroid[axis] += x[axis] / nodes.len() as f64)
        });
        let mut closure = [0.0; 3];
        let mut element_volume = 0.0;
        element_faces.iter().for_each(|&face| {
            let loop_nodes = &faces_nodes[face];
            let mut area = [0.0; 3];
            let mut face_centroid = [0.0; 3];
            (0..loop_nodes.len()).for_each(|k| {
                let a = point(mesh, loop_nodes[k]);
                let b = point(mesh, loop_nodes[(k + 1) % loop_nodes.len()]);
                let c = cross(a, b);
                (0..3).for_each(|axis| {
                    area[axis] += 0.5 * c[axis];
                    face_centroid[axis] += a[axis] / loop_nodes.len() as f64;
                })
            });
            let outward = (0..3)
                .map(|axis| area[axis] * (face_centroid[axis] - centroid[axis]))
                .sum::<f64>();
            let sign = if outward < 0.0 { -1.0 } else { 1.0 };
            (0..3).for_each(|axis| {
                closure[axis] += sign * area[axis];
                element_volume += sign * area[axis] * face_centroid[axis] / 3.0
            })
        });
        closure
            .iter()
            .for_each(|&component| assert!(component.abs() < 1e-9, "not closed: {closure:?}"));
        total += element_volume
    });
    total
}

fn area(mesh: &Mesh<2>) -> f64 {
    let (elements_faces, faces_nodes) = polytopal(mesh);
    let mut total = 0.0;
    elements_faces.iter().for_each(|element_faces| {
        let nodes = element_faces
            .iter()
            .flat_map(|&face| faces_nodes[face].iter().copied())
            .collect::<HashSet<_>>();
        let mut centroid = [0.0; 2];
        nodes.iter().for_each(|&node| {
            let x = point(mesh, node);
            (0..2).for_each(|axis| centroid[axis] += x[axis] / nodes.len() as f64)
        });
        let mut closure = [0.0; 2];
        let mut element_area = 0.0;
        element_faces.iter().for_each(|&face| {
            let a = point(mesh, faces_nodes[face][0]);
            let b = point(mesh, faces_nodes[face][1]);
            let normal = [b[1] - a[1], a[0] - b[0]];
            let midpoint = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let outward = (0..2)
                .map(|axis| normal[axis] * (midpoint[axis] - centroid[axis]))
                .sum::<f64>();
            let sign = if outward < 0.0 { -1.0 } else { 1.0 };
            (0..2).for_each(|axis| {
                closure[axis] += sign * normal[axis];
                element_area += sign * normal[axis] * midpoint[axis] / 2.0
            })
        });
        closure
            .iter()
            .for_each(|&component| assert!(component.abs() < 1e-9, "not closed: {closure:?}"));
        total += element_area
    });
    total
}

fn node_at<const D: usize>(mesh: &Mesh<D>, key: [f64; D]) -> usize {
    (0..mesh.coordinates().len())
        .find(|&node| {
            let x = point(mesh, node);
            (0..D).all(|axis| (x[axis] - key[axis]).abs() < 1e-12)
        })
        .unwrap()
}

#[test]
fn octree_single_leaf() {
    let mesh = Mesh::from(octree(1));
    let (elements_faces, faces_nodes) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 1);
    assert_eq!(faces_nodes.len(), 6);
    assert_eq!(mesh.coordinates().len(), 8);
    faces_nodes
        .iter()
        .for_each(|face| assert_eq!(face.len(), 4));
    conformal(&mesh);
    assert!((volume(&mesh) - 1.0).abs() < 1e-12);
}

#[test]
fn octree_uniform() {
    let mut tree = octree(2);
    tree.subdivide(0).unwrap();
    let mesh = Mesh::from(tree);
    let (elements_faces, faces_nodes) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 8);
    assert_eq!(faces_nodes.len(), 36);
    assert_eq!(mesh.coordinates().len(), 27);
    elements_faces
        .iter()
        .for_each(|element| assert_eq!(element.len(), 6));
    faces_nodes
        .iter()
        .for_each(|face| assert_eq!(face.len(), 4));
    conformal(&mesh);
    assert!((volume(&mesh) - 8.0).abs() < 1e-12);
}

#[test]
fn octree_unbalanced() {
    let mut tree = octree(4);
    tree.subdivide(0).unwrap();
    tree.subdivide(1).unwrap();
    let mesh = Mesh::from(tree);
    let (elements_faces, faces_nodes) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 15);
    assert_eq!(faces_nodes.len(), 66);
    assert_eq!(mesh.coordinates().len(), 46);
    let pentagons = faces_nodes.iter().filter(|face| face.len() == 5).count();
    assert_eq!(pentagons, 12);
    assert_eq!(
        faces_nodes.iter().filter(|face| face.len() == 4).count(),
        54
    );
    let coarse_next_to_fine = [0, 1, 3];
    let coarse_away_from_fine = [2, 4, 5, 6];
    coarse_next_to_fine
        .iter()
        .for_each(|&element| assert_eq!(elements_faces[element].len(), 9));
    coarse_away_from_fine
        .iter()
        .for_each(|&element| assert_eq!(elements_faces[element].len(), 6));
    (7..15).for_each(|element| assert_eq!(elements_faces[element].len(), 6));
    let mut expected = vec![
        node_at(&mesh, [2.0, 2.0, 0.0]),
        node_at(&mesh, [4.0, 2.0, 0.0]),
        node_at(&mesh, [4.0, 2.0, 2.0]),
        node_at(&mesh, [2.0, 2.0, 2.0]),
        node_at(&mesh, [2.0, 2.0, 1.0]),
    ];
    expected.sort_unstable();
    assert!(faces_nodes.iter().any(|face| {
        let mut sorted = face.clone();
        sorted.sort_unstable();
        sorted == expected
    }));
    conformal(&mesh);
    assert!((volume(&mesh) - 64.0).abs() < 1e-12);
}

#[test]
fn octree_doubly_unbalanced() {
    let mut tree = octree(8);
    tree.subdivide(0).unwrap();
    tree.subdivide(1).unwrap();
    tree.subdivide(9).unwrap();
    let mesh = Mesh::from(tree);
    let (elements_faces, _) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 22);
    conformal(&mesh);
    assert!((volume(&mesh) - 512.0).abs() < 1e-12);
}

#[test]
fn quadtree_uniform() {
    let mut tree = quadtree(2);
    tree.subdivide(0).unwrap();
    let mesh = Mesh::from(tree);
    let (elements_faces, faces_nodes) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 4);
    assert_eq!(faces_nodes.len(), 12);
    assert_eq!(mesh.coordinates().len(), 9);
    elements_faces
        .iter()
        .for_each(|element| assert_eq!(element.len(), 4));
    faces_nodes
        .iter()
        .for_each(|face| assert_eq!(face.len(), 2));
    conformal(&mesh);
    assert!((area(&mesh) - 4.0).abs() < 1e-12);
}

#[test]
fn quadtree_unbalanced() {
    let mut tree = quadtree(4);
    tree.subdivide(0).unwrap();
    tree.subdivide(1).unwrap();
    let mesh = Mesh::from(tree);
    let (elements_faces, faces_nodes) = polytopal(&mesh);
    assert_eq!(elements_faces.len(), 7);
    assert_eq!(faces_nodes.len(), 20);
    assert_eq!(mesh.coordinates().len(), 14);
    faces_nodes
        .iter()
        .for_each(|face| assert_eq!(face.len(), 2));
    assert_eq!(elements_faces[0].len(), 5);
    assert_eq!(elements_faces[1].len(), 5);
    assert_eq!(elements_faces[2].len(), 4);
    (3..7).for_each(|element| assert_eq!(elements_faces[element].len(), 4));
    conformal(&mesh);
    assert!((area(&mesh) - 16.0).abs() < 1e-12);
}
