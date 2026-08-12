use super::geometry::star_volume;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, PolytopalConnectivity, tessellation::Tessellation},
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{CrossProduct, Tensor},
};
use std::collections::HashMap;

/// Composes the two cut phases, as the callers of the old one-shot entry
/// points did.
fn cut(
    tessellation: &Tessellation,
    balancing: Balancing,
    scale: f64,
) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.dual_background(balancing, scale)?;
    tessellation.cut(mesh, &classes)
}

fn cut_uniform(tessellation: &Tessellation, spacing: f64) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.lattice_background(spacing)?;
    tessellation.cut(mesh, &classes)
}

fn cut_polyhedral(
    tessellation: &Tessellation,
    balancing: Balancing,
    scale: f64,
) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.octree_background(balancing, scale)?;
    tessellation.cut_polyhedral(mesh, &classes)
}

pub(super) fn signed_volumes(
    polyhedra: &PolytopalConnectivity<3>,
    coordinates: &Coordinates<3>,
) -> Vec<f64> {
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

pub(super) fn dual(tessellation: &Tessellation, scale: f64) -> Mesh<3> {
    let mut octree =
        Octree::<u16, usize>::from_features(tessellation, scale, CurvatureSizing::default(), 2);
    octree
        .equilibrate(Balancing::Strong(1), Pairing::Regular)
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

pub(super) fn sphere(refinements: usize) -> Tessellation {
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

pub(super) fn box_surface(minimum: [f64; 3], maximum: [f64; 3]) -> Tessellation {
    let [x0, y0, z0] = minimum;
    let [x1, y1, z1] = maximum;
    let coordinates = vec![
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ];
    let quads: [[usize; 4]; 6] = [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ];
    let faces: Vec<[usize; 3]> = quads
        .iter()
        .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

pub(super) fn hexahedron(minimum: [f64; 3], maximum: [f64; 3]) -> Mesh<3> {
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
fn cut_sphere() {
    let tessellation = sphere(3);
    let mesh = cut(&tessellation, Balancing::Strong(1), 8.0).unwrap();
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
                let norm = coordinates[node].norm().value();
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
fn cut_thin_plate() {
    let plate = box_surface([-2.0, -2.0, -0.05], [2.0, 2.0, 0.05]);
    let mesh = cut(&plate, Balancing::Strong(1), 4.0);
    assert!(mesh.is_ok(), "{}", mesh.err().unwrap_or(""));
}

#[test]
fn cut_polyhedral_sphere() {
    let tessellation = sphere(2);
    let exact: f64 = tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| {
            let coordinates = tessellation.mesh().coordinates();
            (coordinates[triangle[0]].cross(&coordinates[triangle[1]]) * &coordinates[triangle[2]])
                / 6.0
        })
        .sum();
    [
        Balancing::Strong(1),
        Balancing::Strong(2),
        Balancing::Weak(1),
        Balancing::Weak(3),
    ]
    .into_iter()
    .for_each(|balancing| {
        let mesh = cut_polyhedral(&tessellation, balancing, 4.0).unwrap();
        assert_eq!(mesh.number_of_element_blocks(), 1);
        match &mesh.connectivities()[0] {
            Connectivity::Polyhedral(connectivity) => {
                connectivity
                    .elements_faces()
                    .iter()
                    .for_each(|faces| assert!(faces.len() > 3));
                let volumes = signed_volumes(connectivity, mesh.coordinates());
                volumes
                    .iter()
                    .for_each(|volume| assert!(*volume > 0.0, "{volume}"));
                let volume: f64 = volumes.iter().sum();
                assert!(
                    (volume - exact).abs() / exact < 0.03,
                    "{balancing:?}: {volume} vs {exact}"
                )
            }
            _ => panic!("expected a polyhedral mesh"),
        }
    })
}

#[test]
fn cut_uniform_sphere() {
    let tessellation = sphere(3);
    let mesh = cut_uniform(&tessellation, 0.15).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 2);
    let coordinates = mesh.coordinates();
    match &mesh.connectivities()[1] {
        Connectivity::Polyhedral(polyhedra) => signed_volumes(polyhedra, coordinates)
            .iter()
            .for_each(|&volume| assert!(volume > 0.0, "{volume}")),
        _ => panic!(),
    }
}

#[test]
fn cut_uniform_thin_plate() {
    let plate = box_surface([-2.0, -2.0, -0.05], [2.0, 2.0, 0.05]);
    let mesh = cut_uniform(&plate, 0.25);
    assert!(mesh.is_ok(), "{}", mesh.err().unwrap_or(""));
}

fn rotated_box(minimum: [f64; 3], maximum: [f64; 3], angle: f64) -> Tessellation {
    let plain = box_surface(minimum, maximum);
    let axis = [1.0 / 3.0_f64.sqrt(); 3];
    let (sin, cos) = angle.sin_cos();
    let rotate = |p: [f64; 3]| {
        let dot = (0..3).map(|d| p[d] * axis[d]).sum::<f64>();
        let cross = [
            axis[1] * p[2] - axis[2] * p[1],
            axis[2] * p[0] - axis[0] * p[2],
            axis[0] * p[1] - axis[1] * p[0],
        ];
        std::array::from_fn(|d| p[d] * cos + cross[d] * sin + axis[d] * dot * (1.0 - cos))
    };
    let triangles = match &plain.mesh().connectivities()[0] {
        Connectivity::Triangular(triangles) => triangles.iter().copied().collect::<Vec<_>>(),
        _ => panic!(),
    };
    let coordinates: Vec<[f64; 3]> = plain
        .mesh()
        .coordinates()
        .iter()
        .map(|p| rotate([p[0], p[1], p[2]]))
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        Coordinates::from(coordinates),
    )))
}

fn corners_landed_on(tessellation: &Tessellation, mesh: &Mesh<3>) -> usize {
    tessellation
        .features()
        .corners()
        .iter()
        .filter(|corner| {
            mesh.coordinates()
                .iter()
                .any(|point| (point - *corner).norm() < 1.0e-9)
        })
        .count()
}

#[test]
fn snapping_lands_nodes_on_the_corners_of_an_off_axis_box() {
    let tessellation = rotated_box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 0.7);
    let mesh = cut_uniform(&tessellation, 0.14).unwrap();
    assert_eq!(tessellation.features().corners().len(), 8);
    assert!(corners_landed_on(&tessellation, &mesh) >= 5);
}

#[test]
fn a_corner_takes_at_most_one_node() {
    let tessellation = rotated_box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], 0.7);
    let mesh = cut_uniform(&tessellation, 0.14).unwrap();
    tessellation.features().corners().iter().for_each(|corner| {
        let landed = mesh
            .coordinates()
            .iter()
            .filter(|point| (*point - corner).norm() < 1.0e-9)
            .count();
        assert!(landed <= 1, "{landed} nodes on {corner}")
    })
}
