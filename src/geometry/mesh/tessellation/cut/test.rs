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

/// A stellated sphere: every triangle is replaced by a spike raised from its
/// centroid, giving ridges and points that no smooth surface has.
pub(super) fn star(refinements: usize, height: f64) -> Tessellation {
    let base = sphere(refinements);
    let coordinates_base = base.mesh().coordinates();
    let mut coordinates: Vec<[f64; 3]> = coordinates_base
        .iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect();
    let mut faces = Vec::new();
    base.mesh()
        .connectivities()
        .iter()
        .flatten()
        .for_each(|triangle| {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
            let centroid: Vec<f64> = (0..3)
                .map(|d| (coordinates[a][d] + coordinates[b][d] + coordinates[c][d]) / 3.0)
                .collect();
            let norm = centroid
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            coordinates.push([
                centroid[0] / norm * height,
                centroid[1] / norm * height,
                centroid[2] / norm * height,
            ]);
            let apex = coordinates.len() - 1;
            faces.push([a, b, apex]);
            faces.push([b, c, apex]);
            faces.push([c, a, apex]);
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

pub(super) fn rotated(tessellation: &Tessellation, angles: [f64; 3]) -> Tessellation {
    let [x, y, z] = angles;
    let rotate = |point: &Coordinate<3>| {
        let (a, b, c) = (point[0], point[1], point[2]);
        let (b, c) = (b * x.cos() - c * x.sin(), b * x.sin() + c * x.cos());
        let (a, c) = (a * y.cos() + c * y.sin(), -a * y.sin() + c * y.cos());
        let (a, b) = (a * z.cos() - b * z.sin(), a * z.sin() + b * z.cos());
        [a, b, c]
    };
    let coordinates: Vec<[f64; 3]> = tessellation
        .mesh()
        .coordinates()
        .iter()
        .map(rotate)
        .collect();
    let faces: Vec<[usize; 3]> = tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| [triangle[0], triangle[1], triangle[2]])
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

pub(super) fn shifted(tessellation: &Tessellation, offset: [f64; 3]) -> Tessellation {
    let coordinates: Vec<[f64; 3]> = tessellation
        .mesh()
        .coordinates()
        .iter()
        .map(|point| {
            [
                point[0] + offset[0],
                point[1] + offset[1],
                point[2] + offset[2],
            ]
        })
        .collect();
    let faces: Vec<[usize; 3]> = tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| [triangle[0], triangle[1], triangle[2]])
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
    let mesh = tessellation.cut(Balancing::Strong(1), 8.0).unwrap();
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
fn cut_thin_plate() {
    let plate = box_surface([-2.0, -2.0, -0.05], [2.0, 2.0, 0.05]);
    let mesh = plate.cut(Balancing::Strong(1), 4.0);
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
        let mesh = tessellation.cut_polyhedral(balancing, 4.0).unwrap();
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
    let mesh = tessellation.cut_uniform(0.15).unwrap();
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
    let mesh = plate.cut_uniform(0.25);
    assert!(mesh.is_ok(), "{}", mesh.err().unwrap_or(""));
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture --test-threads=1"]
fn bone_uniform() {
    use crate::geometry::mesh::quality::metrics::Verdict;
    use std::{path::Path, time::Instant};
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    let coordinates = tessellation.mesh().coordinates();
    let bounds: Vec<f64> = (0..3)
        .map(|d| {
            let low = coordinates
                .iter()
                .map(|p| p[d])
                .fold(f64::INFINITY, f64::min);
            let high = coordinates
                .iter()
                .map(|p| p[d])
                .fold(f64::NEG_INFINITY, f64::max);
            high - low
        })
        .collect();
    println!(
        "bone: {} triangles, extent {:.1} x {:.1} x {:.1}",
        tessellation.mesh().number_of_elements(),
        bounds[0],
        bounds[1],
        bounds[2]
    );
    println!(
        "{:>8}  {:>8}  {:>8}  {:>7}  {:>8}  {:>8}  {:>6}  {:>8}  {:>8}",
        "spacing", "hexes", "polys", "cut s", "min SJ", "min vol", "bad", "certif", "margin"
    );
    let longest = bounds.iter().cloned().fold(0.0, f64::max);
    for divisions in [16.0, 32.0, 64.0, 128.0] {
        let spacing = longest / divisions;
        let start = Instant::now();
        match tessellation.cut_uniform(spacing) {
            Err(error) => println!("{spacing:>8.3}  {error}"),
            Ok(mesh) => {
                let seconds = start.elapsed().as_secs_f64();
                // Verdict covers no polyhedra, so the cut cells are judged by
                // signed volume and only the hexes by scaled Jacobian.
                let (block, polyhedra, volumes) =
                    match (&mesh.connectivities()[0], &mesh.connectivities()[1]) {
                        (Connectivity::Hexahedral(hexes), Connectivity::Polyhedral(polyhedra)) => (
                            hexes.iter().copied().collect::<Vec<_>>(),
                            polyhedra.elements_faces().len(),
                            signed_volumes(polyhedra, mesh.coordinates()),
                        ),
                        _ => panic!(),
                    };
                let hexes = block.len();
                let only = Mesh::from((
                    vec![Connectivity::Hexahedral(block.into())],
                    mesh.coordinates().clone(),
                ));
                let scaled = &only.minimum_scaled_jacobians()[0];
                let (certified, worst) = match &only.connectivities()[0] {
                    Connectivity::Hexahedral(hexes) => {
                        hexes
                            .iter()
                            .fold((0usize, f64::INFINITY), |(count, worst), hex| {
                                use crate::geometry::mesh::quality::metrics::hexahedron::bernstein;
                                let element: Vec<usize> = hex.to_vec();
                                (
                                    count
                                        + bernstein::certifies(&element, only.coordinates())
                                            as usize,
                                    worst.min(bernstein::margin(&element, only.coordinates())),
                                )
                            })
                    }
                    _ => panic!(),
                };
                let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
                let volume = volumes.iter().cloned().fold(f64::INFINITY, f64::min);
                let bad = volumes.iter().filter(|&&v| v <= 0.0).count()
                    + scaled.iter().filter(|&&v| v <= 0.0).count();
                println!(
                    "{spacing:>8.3}  {hexes:>8}  {polyhedra:>8}  {seconds:>7.2}  {minimum:>8.4}  {volume:>8.2e}  {bad:>6}  {:>7.1}%  {worst:>8.4}",
                    100.0 * certified as f64 / hexes as f64,
                );
                use crate::{
                    geometry::mesh::{Output, Vtk},
                    io::{Write, write::Compression},
                };
                let mut regular = 0;
                let mut total = 0;
                if let Connectivity::Polyhedral(cells) = &mesh.connectivities()[1] {
                    let faces_nodes = cells.faces_nodes();
                    cells.elements_faces().iter().for_each(|faces| {
                        let mut degree: HashMap<usize, std::collections::HashSet<[usize; 2]>> =
                            HashMap::new();
                        faces.iter().for_each(|&face| {
                            let nodes = &faces_nodes[face];
                            (0..nodes.len()).for_each(|i| {
                                let (a, b) = (nodes[i], nodes[(i + 1) % nodes.len()]);
                                let edge = if a < b { [a, b] } else { [b, a] };
                                degree.entry(a).or_default().insert(edge);
                                degree.entry(b).or_default().insert(edge);
                            })
                        });
                        total += 1;
                        regular += degree.values().all(|edges| edges.len() == 3) as usize;
                    })
                }
                println!(
                    "          {regular}/{total} cut cells are 3-regular ({:.1}%)",
                    100.0 * regular as f64 / total as f64
                );
                let path = format!("bone_uniform_{divisions:.0}.vtm");
                mesh.write(Output::Vtk(Vtk::MultiBlock(Compression::Off(&path))))
                    .unwrap();
                println!("          wrote {path}");
            }
        }
    }
}

#[test]
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn certification_on_sharp_features() {
    use crate::{
        geometry::mesh::quality::metrics::{Verdict, hexahedron::bernstein},
        io::Write,
    };
    let fixtures: Vec<(&str, Tessellation, f64)> = vec![
        ("box", box_surface([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]), 1.0),
        (
            "box_tilt",
            rotated(
                &box_surface([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
                [0.3, 0.4, 0.5],
            ),
            1.0,
        ),
        (
            "tilt_jit",
            shifted(
                &rotated(
                    &box_surface([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
                    [0.3, 0.4, 0.5],
                ),
                [0.013_717, 0.007_193, 0.002_971],
            ),
            1.0,
        ),
        (
            "box_skew",
            rotated(
                &box_surface([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
                [0.1, 0.02, 0.7],
            ),
            1.0,
        ),
        (
            "slab",
            box_surface([-0.5, -0.5, -0.05], [0.5, 0.5, 0.05]),
            1.0,
        ),
        ("star2", star(1, 2.0), 2.0),
        ("star3", star(2, 1.6), 1.6),
        ("spike", star(1, 4.0), 4.0),
    ];
    println!(
        "{:>8}  {:>6}  {:>8}  {:>8}  {:>7}  {:>8}  {:>8}  {:>8}",
        "fixture", "tris", "spacing", "hexes", "polys", "min SJ", "certif", "margin"
    );
    for (name, tessellation, extent) in fixtures {
        tessellation.write(format!("{name}.stl")).unwrap();
        let triangles = tessellation.mesh().number_of_elements();
        for divisions in [16.0, 48.0] {
            let spacing = 2.0 * extent / divisions;
            match tessellation.cut_uniform(spacing) {
                Err(error) => println!("{name:>8}  {triangles:>6}  {spacing:>8.4}  {error}"),
                Ok(mesh) => {
                    let mut block = Vec::new();
                    let mut polyhedra = 0;
                    mesh.connectivities()
                        .iter()
                        .for_each(|connectivity| match connectivity {
                            Connectivity::Hexahedral(hexes) => block.extend(hexes.iter().copied()),
                            Connectivity::Polyhedral(cells) => {
                                polyhedra += cells.elements_faces().len()
                            }
                            _ => panic!(),
                        });
                    let (certified, worst) =
                        block
                            .iter()
                            .fold((0usize, f64::INFINITY), |(count, worst), hex| {
                                let element = hex.to_vec();
                                (
                                    count
                                        + bernstein::certifies(&element, mesh.coordinates())
                                            as usize,
                                    worst.min(bernstein::margin(&element, mesh.coordinates())),
                                )
                            });
                    let hexes = block.len();
                    let only = Mesh::from((
                        vec![Connectivity::Hexahedral(block.into())],
                        mesh.coordinates().clone(),
                    ));
                    let minimum = only.minimum_scaled_jacobians()[0]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);
                    println!(
                        "{name:>8}  {triangles:>6}  {spacing:>8.4}  {hexes:>8}  {polyhedra:>7}  {minimum:>8.4}  {:>7.1}%  {worst:>8.4}",
                        100.0 * certified as f64 / hexes as f64
                    );
                }
            }
        }
        for scale in [8.0, 16.0] {
            match tessellation.cut(Balancing::Strong(1), scale) {
                Ok(mesh) => println!(
                    "{name:>8}  dual scale {scale:>5}  ok, {} elements",
                    mesh.number_of_elements()
                ),
                Err(error) => println!("{name:>8}  dual scale {scale:>5}  {error}"),
            }
        }
    }
}
