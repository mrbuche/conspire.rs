use super::{Class, geometry::star_volume};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Dualization, Mesh, PolytopalConnectivity, Verdict,
            tessellation::Tessellation,
        },
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{CrossProduct, Quantity, Tensor},
};
use std::collections::HashMap;

fn cut(
    tessellation: &Tessellation,
    balancing: Balancing,
    scale: f64,
) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.dual_background(balancing, scale, None)?;
    tessellation.cut(mesh, &classes)
}

fn cut_uniform(tessellation: &Tessellation, spacing: f64) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.lattice_background(Quantity::new(spacing))?;
    tessellation.cut(mesh, &classes)
}

fn cut_polyhedral(
    tessellation: &Tessellation,
    balancing: Balancing,
    scale: f64,
) -> Result<Mesh<3>, &'static str> {
    let (mesh, classes) = tessellation.octree_background(balancing, scale, None)?;
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

pub(super) fn dual(tessellation: &Tessellation, scale: f64) -> Mesh<3> {
    let mut octree =
        Octree::<u16, usize>::from_features(tessellation, scale, CurvatureSizing::default(), 2)
            .unwrap();
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
    let mesh = cut(&tessellation, Balancing::Strong(1), 12.0).unwrap();
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
                    assert!(volume < star.value() * (1.0 + 1e-9), "{volume} {star}")
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
                .value()
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
        let mesh = cut_polyhedral(&tessellation, balancing, 8.0).unwrap();
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
        .map(|p| rotate([p[0].value(), p[1].value(), p[2].value()]))
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
                .any(|point| (point - *corner).norm() < Quantity::new(1.0e-9))
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
            .filter(|point| (*point - corner).norm() < Quantity::new(1.0e-9))
            .count();
        assert!(landed <= 1, "{landed} nodes on {corner}")
    })
}

fn tets(mesh: &Mesh<3>) -> Vec<[usize; 4]> {
    mesh.connectivities()
        .iter()
        .flat_map(|block| match block {
            Connectivity::Tetrahedral(elements) => elements.iter().copied(),
            _ => panic!("expected a tetrahedral block"),
        })
        .collect()
}

fn worst_scaled_jacobian(mesh: &Mesh<3>) -> f64 {
    mesh.minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(f64::INFINITY, f64::min)
}

fn volume_of(mesh: &Mesh<3>) -> f64 {
    mesh.volumes().into_iter().flatten().sum()
}

fn spanned(mesh: &Mesh<3>) -> f64 {
    let coordinates = mesh.coordinates();
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    (0..mesh.number_of_nodes()).for_each(|node| {
        (0..3).for_each(|axis| {
            low[axis] = low[axis].min(coordinates[node][axis].value());
            high[axis] = high[axis].max(coordinates[node][axis].value())
        })
    });
    (0..3).map(|axis| high[axis] - low[axis]).product()
}

fn boundary_faces(mesh: &Mesh<3>) -> usize {
    let mut faces = HashMap::<[usize; 3], usize>::new();
    tets(mesh).iter().for_each(|tet| {
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            .iter()
            .for_each(|local| {
                let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                face.sort_unstable();
                *faces.entry(face).or_default() += 1
            })
    });
    assert!(
        faces.values().all(|&count| count <= 2),
        "face on three tets"
    );
    faces.values().filter(|&&count| count == 1).count()
}

fn conforming_in_a_box(mesh: &Mesh<3>) {
    let coordinates = mesh.coordinates();
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    (0..mesh.number_of_nodes()).for_each(|node| {
        (0..3).for_each(|axis| {
            low[axis] = low[axis].min(coordinates[node][axis].value());
            high[axis] = high[axis].max(coordinates[node][axis].value())
        })
    });
    let span = (0..3).fold(0.0_f64, |m, axis| m.max(high[axis] - low[axis]));
    let on_hull = |face: &[usize; 3]| {
        (0..3).any(|axis| {
            let flush = |bound: f64| {
                face.iter()
                    .all(|&node| (coordinates[node][axis].value() - bound).abs() < 1.0e-9 * span)
            };
            flush(low[axis]) || flush(high[axis])
        })
    };
    let mut faces = HashMap::<[usize; 3], usize>::new();
    tets(mesh).iter().for_each(|tet| {
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            .iter()
            .for_each(|local| {
                let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                face.sort_unstable();
                *faces.entry(face).or_default() += 1
            })
    });
    assert!(faces.values().all(|&count| count <= 2));
    assert_eq!(
        faces
            .iter()
            .filter(|&(face, &count)| count == 1 && !on_hull(face))
            .count(),
        0,
        "non-conforming: interior face used once"
    )
}

#[test]
fn lattice_tet_background_is_six_tets_per_cell() {
    let tessellation = sphere(3);
    let spacing = Quantity::new(0.15);
    let (hexes, cells) = tessellation.lattice_background(spacing).unwrap();
    let (mesh, classes) = tessellation.lattice_tet_background(spacing).unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(mesh.number_of_elements(), 6 * hexes.number_of_elements());
    assert_eq!(classes.len(), mesh.number_of_elements());
    assert_eq!(cells.len(), hexes.number_of_elements());
    cells
        .iter()
        .zip(classes.chunks(6))
        .for_each(|(&cell, tets)| assert!(tets.iter().all(|&class| class == cell)));
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
    let cubes = volume_of(&hexes);
    assert!(
        (volume_of(&mesh) - cubes).abs() < 1.0e-9 * cubes,
        "{} vs {cubes}",
        volume_of(&mesh)
    );
    assert_eq!(boundary_faces(&mesh), 2 * hexes.exterior_faces().len())
}

#[test]
fn classify_agrees_between_the_hex_and_tet_lattices() {
    let tessellation = sphere(3);
    let spacing = Quantity::new(0.2);
    let (hexes, cells) = tessellation.lattice_background(spacing).unwrap();
    let (mesh, classes) = tessellation.lattice_tet_background(spacing).unwrap();
    let found = tessellation.classify(&mesh);
    let hex_found = tessellation.classify(&hexes);
    assert_eq!(found.len(), mesh.number_of_elements());
    assert_eq!(hex_found.len(), hexes.number_of_elements());
    hex_found
        .iter()
        .zip(found.chunks(6))
        .for_each(|(&cell, tets)| assert!(tets.iter().all(|&class| class == cell)));
    let cells_differing: std::collections::BTreeSet<usize> = classes
        .iter()
        .zip(&found)
        .enumerate()
        .filter(|&(_, (&rasterized, &found))| rasterized != found)
        .map(|(element, _)| element / 6)
        .collect();
    let hexes_differing: std::collections::BTreeSet<usize> = cells
        .iter()
        .zip(&hex_found)
        .enumerate()
        .filter(|&(_, (&rasterized, &found))| rasterized != found)
        .map(|(element, _)| element)
        .collect();
    assert_eq!(cells_differing, hexes_differing);
    assert!(classes.contains(&Class::Cut));
    assert!(found.contains(&Class::Inside));
    assert!(found.contains(&Class::Outside))
}

fn check_octree_tet_background(tessellation: &Tessellation, graded: bool) {
    let (mesh, classes) = tessellation
        .octree_tet_background(Balancing::Strong(1), Pairing::Regular, 1.0, None)
        .unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(classes.len(), mesh.number_of_elements());
    assert!(worst_scaled_jacobian(&mesh) > 0.0);
    conforming_in_a_box(&mesh);
    let root = spanned(&mesh);
    assert!(
        (volume_of(&mesh) - root).abs() < 1.0e-9 * root,
        "{} vs {root}",
        volume_of(&mesh)
    );
    let (polyhedra, _) = tessellation
        .octree_background(Balancing::Strong(1), 1.0, None)
        .unwrap();
    let leaves = polyhedra.number_of_elements();
    if graded {
        assert!(
            (6 * leaves..=14 * leaves).contains(&mesh.number_of_elements()),
            "{} over {leaves} leaves",
            mesh.number_of_elements()
        );
        assert!(mesh.number_of_elements() > 6 * leaves, "no graded leaf")
    } else {
        assert_eq!(mesh.number_of_elements(), 6 * leaves)
    }
}

#[test]
fn octree_tet_background_of_a_sphere_is_uniform() {
    check_octree_tet_background(&sphere(3), false)
}

#[test]
fn octree_tet_background_of_a_slab_is_conforming() {
    check_octree_tet_background(&box_surface([-2.0, -1.0, -0.15], [2.0, 1.0, 0.15]), true)
}

#[test]
fn octree_tet_background_requires_strong_1() {
    let tessellation = sphere(2);
    assert!(
        tessellation
            .octree_tet_background(Balancing::Weak(1), Pairing::Regular, 1.0, None)
            .is_err()
    );
    assert!(
        tessellation
            .octree_tet_background(Balancing::Strong(2), Pairing::Regular, 1.0, None)
            .is_err()
    );
    assert!(
        tessellation
            .octree_tet_background(Balancing::Strong(1), Pairing::Regular, 1.0, None)
            .is_ok()
    )
}
