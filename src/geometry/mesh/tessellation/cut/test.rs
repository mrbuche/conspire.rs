use super::{Class, Sign, contained, star_volume};
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, PolytopalConnectivity, tessellation::Tessellation},
        ntree::{Balance, Balancing, Dualization, Octree, Pairing},
    },
    math::{CrossProduct, Tensor},
};
use std::collections::{HashMap, HashSet};

fn signed_volumes(polyhedra: &PolytopalConnectivity<3>, coordinates: &Coordinates<3>) -> Vec<f64> {
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
                        .sum::<crate::geometry::Coordinate<3>>()
                        / nodes.len() as f64;
                    let volume: f64 = (0..nodes.len())
                        .map(|i| {
                            let one = &coordinates[nodes[i]];
                            let two = &coordinates[nodes[(i + 1) % nodes.len()]];
                            &middle * &one.cross(two) / 6.0
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

fn dual(tessellation: &Tessellation, scale: f64) -> Mesh<3> {
    let mut octree = Octree::<u16, usize>::from_sdf(tessellation, scale, 2);
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    octree.dualize()
}

fn midpoint(
    a: usize,
    b: usize,
    coordinates: &mut Vec<[f64; 3]>,
    cache: &mut HashMap<[usize; 2], usize>,
) -> usize {
    let key = if a < b { [a, b] } else { [b, a] };
    *cache.entry(key).or_insert_with(|| {
        let (p, q) = (coordinates[a], coordinates[b]);
        let m = [p[0] + q[0], p[1] + q[1], p[2] + q[2]];
        let norm = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        coordinates.push([m[0] / norm, m[1] / norm, m[2] / norm]);
        coordinates.len() - 1
    })
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
                let ab = midpoint(a, b, &mut coordinates, &mut cache);
                let bc = midpoint(b, c, &mut coordinates, &mut cache);
                let ca = midpoint(c, a, &mut coordinates, &mut cache);
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect()
    });
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn hexahedron(minimum: [f64; 3], maximum: [f64; 3]) -> Mesh<3> {
    let [x0, y0, z0] = minimum;
    let [x1, y1, z1] = maximum;
    Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        Coordinates::from(vec![
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]),
    ))
}

#[test]
fn classify_single_hexahedra() {
    let tessellation = sphere(3);
    assert_eq!(
        tessellation.classify(&hexahedron([-0.1; 3], [0.1; 3])),
        vec![Class::Inside]
    );
    assert_eq!(
        tessellation.classify(&hexahedron([2.0; 3], [3.0; 3])),
        vec![Class::Outside]
    );
    assert_eq!(
        tessellation.classify(&hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1])),
        vec![Class::Cut]
    );
}

#[test]
fn containment() {
    let tessellation = sphere(3);
    let straddling = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    assert!(!contained(&straddling, &tessellation.classify(&straddling)));
    let enclosing = hexahedron([-2.0; 3], [2.0; 3]);
    assert!(!contained(&enclosing, &tessellation.classify(&enclosing)));
    let outside = hexahedron([2.0; 3], [3.0; 3]);
    assert!(contained(&outside, &tessellation.classify(&outside)))
}

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
    tables
        .crossings()
        .values()
        .for_each(|point| assert!((point.norm() - 1.0).abs() < 0.01));
    assert_eq!(tables.segments().len(), 4);
    tables
        .segments()
        .values()
        .for_each(|segments| assert_eq!(segments.len(), 1))
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
    tables.crossings().values().for_each(|point| {
        let norm = point.norm();
        assert!((0.985..=1.0 + 1e-9).contains(&norm), "{norm}")
    });
    let coordinates = mesh.coordinates();
    tables.signs().iter().for_each(|(&node, &sign)| {
        let norm = coordinates[node].norm();
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
        segment
            .iter()
            .for_each(|edge| assert!(edge[0] == edge[1] || tables.crossings().contains_key(edge)))
    })
}

#[test]
fn classify_sphere_dual() {
    let tessellation = sphere(3);
    let mesh = dual(&tessellation, 8.0);
    let classes = tessellation.classify(&mesh);
    [Class::Inside, Class::Cut, Class::Outside]
        .iter()
        .for_each(|class| assert!(classes.contains(class)));
    let centroids = mesh.centroids();
    classes
        .iter()
        .zip(centroids.iter())
        .for_each(|(class, centroid)| match class {
            Class::Inside => assert!(centroid.norm() < 1.0),
            Class::Outside => assert!(centroid.norm() > 1.0),
            Class::Cut => (),
        });
    let mut faces: HashMap<Vec<usize>, Vec<Class>> = HashMap::new();
    mesh.iter().for_each(|block| {
        block
            .iter()
            .zip(classes.iter())
            .for_each(|(element, &class)| {
                block.local_faces().iter().for_each(|face| {
                    let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                    key.sort_unstable();
                    faces.entry(key).or_default().push(class);
                })
            })
    });
    faces.values().for_each(|classes| {
        assert!(!(classes.contains(&Class::Inside) && classes.contains(&Class::Outside)))
    })
}

#[test]
fn snap_eliminates_sliver() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.95, -0.1, -0.1], [1.15, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let (mesh, snapped) = tessellation.snap(mesh, &classes).unwrap();
    assert_eq!(snapped.len(), 4);
    let coordinates = mesh.coordinates();
    snapped
        .iter()
        .for_each(|&node| assert!((coordinates[node].norm() - 1.0).abs() < 0.01));
    let tables = tessellation.tables(&mesh, &classes, &snapped).unwrap();
    assert!(tables.crossings().is_empty());
    let result = tessellation.assemble(&mesh, &classes, &tables).unwrap();
    assert_eq!(result.number_of_element_blocks(), 0)
}

#[test]
fn assemble_single_hexahedron() {
    let tessellation = sphere(3);
    let mesh = hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1]);
    let classes = tessellation.classify(&mesh);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
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
            assert!(base > 0.0)
        }
        _ => panic!(),
    }
}

#[test]
fn cut_sphere() {
    let tessellation = sphere(3);
    let mesh = tessellation.cut(Balancing::Strong, 8.0).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 2);
    let coordinates = mesh.coordinates();
    let mut usage: HashMap<Vec<usize>, usize> = HashMap::new();
    mesh.iter().for_each(|block| match block {
        Connectivity::Hexahedral(_) => block.iter().for_each(|element| {
            block.local_faces().iter().for_each(|face| {
                let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                key.sort_unstable();
                *usage.entry(key).or_insert(0) += 1;
            })
        }),
        Connectivity::Polyhedral(polyhedra) => {
            polyhedra.elements_faces().iter().for_each(|faces| {
                faces.iter().for_each(|&face| {
                    let mut key = polyhedra.faces_nodes()[face].clone();
                    key.sort_unstable();
                    *usage.entry(key).or_insert(0) += 1;
                })
            })
        }
        _ => panic!(),
    });
    usage.values().for_each(|&count| assert!(count <= 2));
    usage
        .iter()
        .filter(|&(_, &count)| count == 1)
        .for_each(|(key, _)| {
            key.iter().for_each(|&node| {
                let norm = coordinates[node].norm();
                assert!((0.985..=1.0 + 1e-9).contains(&norm), "{norm}")
            })
        });
    match &mesh.connectivities()[1] {
        Connectivity::Polyhedral(polyhedra) => {
            let signed = signed_volumes(polyhedra, coordinates);
            signed.iter().for_each(|&volume| assert!(volume > 0.0));
            let faces_nodes = polyhedra.faces_nodes();
            polyhedra
                .elements_faces()
                .iter()
                .zip(signed.iter())
                .for_each(|(faces, &volume)| {
                    let polygons: Vec<Vec<usize>> = faces
                        .iter()
                        .map(|&face| faces_nodes[face].clone())
                        .collect();
                    let star = star_volume(&polygons, coordinates);
                    assert!(volume < star * (1.0 + 1e-9), "{volume} {star}")
                })
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
    let tessellation = Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    )));
    let mesh = Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]].into(),
        )],
        Coordinates::from(vec![
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
        ]),
    ));
    let classes = tessellation.classify(&mesh);
    assert_eq!(classes, vec![Class::Cut, Class::Cut]);
    let tables = tessellation
        .tables(&mesh, &classes, &HashSet::new())
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
            assert!((volume - 0.220095389507154).abs() < 1e-12, "{volume}");
            let signed = signed_volumes(polyhedra, result.coordinates())[0];
            assert!((signed - 0.220095389507154).abs() < 1e-12, "{signed}")
        }
        _ => panic!(),
    }
}
