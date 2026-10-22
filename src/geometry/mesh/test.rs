use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::assert::AssertionError,
};
use std::{
    collections::HashMap,
    f64::consts::{PI, TAU},
};

pub const CONNECTIVITY: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 6, 2],
    [3, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

pub const COORDINATES: [Coordinate<3>; 8] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 1.0, 1.0]),
    Coordinate::const_from([0.0, 1.0, 1.0]),
];

pub fn mesh() -> Mesh<3> {
    let connectivities = vec![Connectivity::Triangular(CONNECTIVITY.to_vec().into())];
    let coordinates = Coordinates::from(COORDINATES);
    (connectivities, coordinates).into()
}

pub fn sphere(stacks: usize, slices: usize, radius: f64) -> Tessellation {
    let mut points = vec![[0.0, 0.0, radius]];
    for i in 1..=stacks {
        let theta = PI * i as f64 / (stacks + 1) as f64;
        for j in 0..slices {
            let phi = TAU * j as f64 / slices as f64;
            points.push([
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            ]);
        }
    }
    let south = points.len();
    points.push([0.0, 0.0, -radius]);
    let ring_start = |i: usize| 1 + (i - 1) * slices;
    let mut faces = Vec::new();
    for j in 0..slices {
        faces.push([0, ring_start(1) + j, ring_start(1) + (j + 1) % slices]);
    }
    for i in 1..stacks {
        for j in 0..slices {
            let (a, b) = (ring_start(i) + j, ring_start(i + 1) + j);
            let (c, d) = (
                ring_start(i + 1) + (j + 1) % slices,
                ring_start(i) + (j + 1) % slices,
            );
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    for j in 0..slices {
        faces.push([
            south,
            ring_start(stacks) + (j + 1) % slices,
            ring_start(stacks) + j,
        ]);
    }
    let coordinates = Coordinates::from(points);
    let connectivities = vec![Connectivity::Triangular(faces.into())];
    Mesh::from((connectivities, coordinates)).into()
}

pub fn mesh_with_node_sets() -> Mesh<3> {
    let mut mesh = mesh();
    mesh.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    mesh
}

#[test]
fn connectivity_coordinates() -> Result<(), AssertionError> {
    let _ = mesh();
    Ok(())
}

// #[test]
// fn connectivity_coordinates_ref() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((connectivity, &coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates_ref() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, &coordinates));
//     Ok(())
// }

pub fn perpendicular_facet(axis: usize, sign: f64, size: f64) -> Mesh<3> {
    let curve = 1.0 - 0.5 * size * size;
    let place = |point: [f64; 3]| -> [f64; 3] {
        let mut placed = [0.0; 3];
        (0..3).for_each(|i| placed[(i + axis) % 3] = sign * point[i]);
        placed
    };
    let coordinates = Coordinates::from(
        [
            [1.0, 0.0, 0.0],
            [curve, size, 0.0],
            [curve, 0.0, size],
            [-1.0, -20.0, -20.0],
            [-1.0, 20.0, -20.0],
            [-1.0, 0.0, 20.0],
        ]
        .map(place)
        .to_vec(),
    );
    let facet = if sign > 0.0 { [0, 1, 2] } else { [0, 2, 1] };
    (
        vec![Connectivity::Triangular(vec![facet, [3, 4, 5]].into())],
        coordinates,
    )
        .into()
}

pub fn octahedron(levels: usize) -> Tessellation {
    let mut points = vec![
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
    for _ in 0..levels {
        let mut midpoints = HashMap::new();
        let mut midpoint = |a: usize, b: usize, points: &mut Vec<[f64; 3]>| {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let sum = [0, 1, 2].map(|i| points[a][i] + points[b][i]);
                let norm = sum.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
                points.push(sum.map(|entry| entry / norm));
                points.len() - 1
            })
        };
        faces = faces
            .into_iter()
            .flat_map(|[a, b, c]| {
                let (ab, bc, ca) = (
                    midpoint(a, b, &mut points),
                    midpoint(b, c, &mut points),
                    midpoint(c, a, &mut points),
                );
                [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            })
            .collect();
    }
    let coordinates = Coordinates::from(points);
    let connectivities = vec![Connectivity::Triangular(faces.into())];
    Mesh::from((connectivities, coordinates)).into()
}

mod node_element_connectivity {
    use crate::geometry::mesh::Connectivity;

    #[test]
    fn block_sized_to_the_number_of_nodes() {
        let block = Connectivity::Triangular(vec![[0_usize, 1, 2]].into());
        let incidence = block.node_element_connectivity(5);
        assert_eq!(incidence.as_slice().len(), 5);
        assert_eq!(incidence.as_slice()[4], Vec::<usize>::new());
        assert_eq!(incidence.number_of_incident_vertices(), 1);
    }
}

/// Triangles, as indices into `outline`, that fill a simple counterclockwise
/// polygon, found by clipping its ears so that it may be non-convex.
fn ears(outline: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    let mut rest: Vec<usize> = (0..outline.len()).collect();
    while rest.len() > 3 {
        let ear = (0..rest.len())
            .find(|&k| {
                let (a, b, c) = (
                    rest[(k + rest.len() - 1) % rest.len()],
                    rest[k],
                    rest[(k + 1) % rest.len()],
                );
                cross(outline[a], outline[b], outline[c]) > 0.0
                    && rest
                        .iter()
                        .filter(|&&p| p != a && p != b && p != c)
                        .all(|&p| {
                            [(a, b), (b, c), (c, a)]
                                .iter()
                                .any(|&(u, v)| cross(outline[u], outline[v], outline[p]) <= 0.0)
                        })
            })
            .expect("outline is simple and counterclockwise");
        triangles.push([
            rest[(ear + rest.len() - 1) % rest.len()],
            rest[ear],
            rest[(ear + 1) % rest.len()],
        ]);
        rest.remove(ear);
    }
    triangles.push([rest[0], rest[1], rest[2]]);
    triangles
}

/// A closed solid extruded one unit along x from a counterclockwise outline
/// in (y, z), its end caps ear-clipped so the outline may be non-convex.
pub fn extrusion(outline: &[[f64; 2]]) -> Tessellation {
    let count = outline.len();
    let points: Vec<[f64; 3]> = [0.0, 1.0]
        .into_iter()
        .flat_map(|x| outline.iter().map(move |&[y, z]| [x, y, z]))
        .collect();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    (0..count).for_each(|i| {
        let j = (i + 1) % count;
        faces.extend([[i, j, count + j], [i, count + j, count + i]])
    });
    ears(outline)
        .into_iter()
        .for_each(|[a, b, c]| faces.extend([[count + a, count + b, count + c], [a, c, b]]));
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

/// A 1 x 1 x 0.5 block with a trench `width` wide and `depth` deep cut down
/// from its top, entering from the middle of its y = 0 face and closing in a
/// semicircle. The floor of the trench is a narrow strip, bounded by creases
/// that run smoothly round its end and so make one chain, in a solid that is
/// thick all round it.
pub fn trench(width: f64, depth: f64) -> Tessellation {
    let (height, length, steps) = (0.5, 0.6, 8);
    let (left, right, radius) = (0.5 - width / 2.0, 0.5 + width / 2.0, width / 2.0);
    let floor = height - depth;
    let mut path: Vec<[f64; 2]> = vec![[left, 0.0], [left, length]];
    (1..steps).for_each(|step| {
        let angle = PI - PI * step as f64 / steps as f64;
        path.push([0.5 + radius * angle.cos(), length + radius * angle.sin()])
    });
    path.extend([[right, length], [right, 0.0]]);
    let count = path.len();
    let last = count - 1;
    let mut points: Vec<[f64; 3]> = path.iter().map(|&[x, y]| [x, y, height]).collect();
    points.extend(path.iter().map(|&[x, y]| [x, y, floor]));
    let corner = 2 * count;
    points.extend([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, height],
        [1.0, 0.0, height],
        [1.0, 1.0, height],
        [0.0, 1.0, height],
        [left, 0.0, 0.0],
        [right, 0.0, 0.0],
    ]);
    let top = |i: usize| i;
    let bottom = |i: usize| count + i;
    let (b00, b10, b11, b01) = (corner, corner + 1, corner + 2, corner + 3);
    let (t00, t10, t11, t01) = (corner + 4, corner + 5, corner + 6, corner + 7);
    let (ba, bb) = (corner + 8, corner + 9);
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut fill = |ring: Vec<usize>, flip: bool| {
        let plan: Vec<[f64; 2]> = ring.iter().map(|&i| [points[i][0], points[i][1]]).collect();
        ears(&plan).into_iter().for_each(|[a, b, c]| {
            faces.push(if flip {
                [ring[a], ring[c], ring[b]]
            } else {
                [ring[a], ring[b], ring[c]]
            })
        })
    };
    fill(
        [t00]
            .into_iter()
            .chain((0..count).map(top))
            .chain([t10, t11, t01])
            .collect(),
        false,
    );
    fill((0..count).rev().map(bottom).collect(), false);
    fill(vec![b00, ba, bb, b10, b11, b01], true);
    let mut quad = |a: usize, b: usize, c: usize, d: usize| faces.extend([[a, b, c], [a, c, d]]);
    (0..last).for_each(|i| quad(bottom(i), bottom(i + 1), top(i + 1), top(i)));
    quad(b11, b01, t01, t11);
    quad(b10, b11, t11, t10);
    quad(b01, b00, t00, t01);
    quad(ba, bb, bottom(last), bottom(0));
    [
        [b00, ba, bottom(0), top(0), t00],
        [b10, t10, top(last), bottom(last), bb],
    ]
    .into_iter()
    .for_each(|ring| {
        faces.extend([
            [ring[0], ring[1], ring[2]],
            [ring[0], ring[2], ring[3]],
            [ring[0], ring[3], ring[4]],
        ])
    });
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

/// A plate `height` tall whose plan is a triangle narrowing to a point at the
/// origin, where its two long sides meet at `angle` degrees, `length` away
/// from the short end. The creases along the top of those sides meet at the
/// point with surface between them, and are `2 s tan(angle / 2)` apart at a
/// distance `s` from it.
pub fn wedge(angle: f64, length: f64, height: f64) -> Tessellation {
    let half = length * (0.5 * angle.to_radians()).tan();
    let points = vec![
        [0.0, 0.0, 0.0],
        [length, half, 0.0],
        [length, -half, 0.0],
        [0.0, 0.0, height],
        [length, half, height],
        [length, -half, height],
    ];
    let faces: Vec<[usize; 3]> = vec![
        [0, 1, 2],
        [3, 5, 4],
        [0, 4, 1],
        [0, 3, 4],
        [0, 2, 5],
        [0, 5, 3],
        [1, 5, 2],
        [1, 4, 5],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

#[test]
fn the_trench_is_a_closed_outward_surface() {
    let (width, depth) = (0.05, 0.3);
    let block = trench(width, depth);
    let coordinates = block.mesh().coordinates();
    let triangles: Vec<[usize; 3]> = block
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|element| [element[0], element[1], element[2]])
        .collect();
    let mut edges = HashMap::<[usize; 2], usize>::new();
    triangles.iter().for_each(|&[a, b, c]| {
        [[a, b], [b, c], [c, a]]
            .into_iter()
            .for_each(|edge| *edges.entry(edge).or_default() += 1)
    });
    edges.iter().for_each(|(&[a, b], &count)| {
        assert_eq!(count, 1, "edge {a}-{b} is used {count} times one way");
        assert_eq!(edges.get(&[b, a]), Some(&1), "edge {a}-{b} has no partner");
    });
    let point = |i: usize| -> [f64; 3] { std::array::from_fn(|axis| coordinates[i][axis].value()) };
    let volume: f64 = triangles
        .iter()
        .map(|&[a, b, c]| {
            let (p, q, r) = (point(a), point(b), point(c));
            (p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0])
                + p[2] * (q[0] * r[1] - q[1] * r[0]))
                / 6.0
        })
        .sum();
    let area = width * 0.6 + PI * (width / 2.0).powi(2) / 2.0;
    let expected = 0.5 - area * depth;
    assert!(
        (volume - expected).abs() < 1.0e-3 * expected,
        "{volume} against {expected}"
    );
}
