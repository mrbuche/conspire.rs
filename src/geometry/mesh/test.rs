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

pub fn square(n: usize) -> Mesh<2> {
    let node = |i: usize, j: usize| j * (n + 1) + i;
    let coordinates: Coordinates<2> = (0..=n)
        .flat_map(|j| {
            (0..=n).map(move |i| Coordinate::from([i as f64 / n as f64, j as f64 / n as f64]))
        })
        .collect();
    let triangles: Vec<[usize; 3]> = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            [
                [node(i, j), node(i + 1, j), node(i + 1, j + 1)],
                [node(i, j), node(i + 1, j + 1), node(i, j + 1)],
            ]
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    ))
}

pub fn tetrahedra(n: usize) -> Mesh<3> {
    let h = 1.0 / n as f64;
    let node = |i: usize, j: usize, k: usize| (k * (n + 1) + j) * (n + 1) + i;
    let coordinates: Coordinates<3> = (0..=n)
        .flat_map(|k| {
            (0..=n).flat_map(move |j| {
                (0..=n).map(move |i| Coordinate::from([i as f64 * h, j as f64 * h, k as f64 * h]))
            })
        })
        .collect();
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let tetrahedra: Vec<[usize; 4]> = (0..n)
        .flat_map(|k| (0..n).flat_map(move |j| (0..n).map(move |i| [i, j, k])))
        .flat_map(|cell| orders.map(|order| (cell, order)))
        .map(|(cell, order)| {
            let mut corner = cell;
            let mut nodes = [node(corner[0], corner[1], corner[2]); 4];
            for (step, axis) in order.into_iter().enumerate() {
                corner[axis] += 1;
                nodes[step + 1] = node(corner[0], corner[1], corner[2]);
            }
            nodes
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Tetrahedral(tetrahedra.into())],
        coordinates,
    ))
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
