use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Quantity, Tensor},
    units::Length,
};
use std::{
    array::from_fn,
    collections::HashMap,
    f64::consts::{PI, TAU},
};

fn sphere(stacks: usize, slices: usize, radius: f64) -> Tessellation {
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
    Tessellation::from(Mesh::from((connectivities, coordinates)))
}

fn curvature(tolerance: Quantity<Length>) -> CurvatureSizing {
    CurvatureSizing {
        tolerance: Some(tolerance),
        ..Default::default()
    }
}

#[test]
fn tighter_curvature_tolerance_refines_more() {
    let tessellation = sphere(12, 24, 2.0);
    let scale = 4.0;
    let loose =
        Octree::<u16, usize>::from_features(&tessellation, scale, curvature(Quantity::new(1.0)), 0);
    let medium = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-2)),
        0,
    );
    let tight = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-3)),
        0,
    );
    assert!(medium.len() > loose.len());
    assert!(tight.len() > medium.len());
}

#[test]
fn default_curvature_sizing_disables_curvature_refinement() {
    let tessellation = sphere(4, 8, 2.0);
    let scale = 4.0;
    let without = Octree::<u16, usize>::from_features(
        &tessellation,
        scale,
        curvature(Quantity::new(1.0e-3)),
        0,
    );
    let with_default =
        Octree::<u16, usize>::from_features(&tessellation, scale, CurvatureSizing::default(), 0);
    assert!(with_default.len() <= without.len());
}

fn tessellate(points: Vec<[f64; 3]>, faces: Vec<[usize; 3]>) -> Tessellation {
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(points),
    )))
}

fn cube(divisions: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut lookup = HashMap::<[usize; 3], usize>::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut node = |grid: [usize; 3]| {
        *lookup.entry(grid).or_insert_with(|| {
            points.push(from_fn::<_, 3, _>(|axis| {
                grid[axis] as f64 / divisions as f64
            }));
            points.len() - 1
        })
    };
    for axis in 0..3 {
        let (along, across) = ((axis + 1) % 3, (axis + 2) % 3);
        for side in [0, divisions] {
            for i in 0..divisions {
                for j in 0..divisions {
                    let corner = |one: usize, two: usize| {
                        let mut grid = [0; 3];
                        grid[axis] = side;
                        grid[along] = i + one;
                        grid[across] = j + two;
                        grid
                    };
                    let (a, b) = (node(corner(0, 0)), node(corner(1, 0)));
                    let (c, d) = (node(corner(1, 1)), node(corner(0, 1)));
                    if side == 0 {
                        faces.extend([[a, c, b], [a, d, c]])
                    } else {
                        faces.extend([[a, b, c], [a, c, d]])
                    }
                }
            }
        }
    }
    (points, faces)
}

fn pored_cube(radius: f64) -> Tessellation {
    let (mut points, mut faces) = cube(1);
    let pore = sphere(12, 24, radius);
    let offset = points.len();
    pore.mesh()
        .coordinates()
        .iter()
        .for_each(|point| points.push(from_fn::<_, 3, _>(|axis| point[axis].value() + 0.5)));
    pore.mesh()
        .connectivities()
        .iter()
        .flatten()
        .for_each(|element| {
            faces.push([
                element[0] + offset,
                element[2] + offset,
                element[1] + offset,
            ])
        });
    tessellate(points, faces)
}

fn refinement(tessellation: &Tessellation, curvature: CurvatureSizing) -> (f64, Vec<[f64; 3]>) {
    let mut octree = Octree::<u16, usize>::from_features(tessellation, 5.0, curvature, 0);
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let background = octree.dualize();
    let cells: Vec<(f64, [f64; 3])> = background
        .connectivities()
        .iter()
        .flatten()
        .map(|element| {
            let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            element.iter().for_each(|&node| {
                let point = &background.coordinates()[node];
                (0..3).for_each(|axis| {
                    min[axis] = min[axis].min(point[axis].value());
                    max[axis] = max[axis].max(point[axis].value());
                })
            });
            let size = (0..3)
                .map(|axis| max[axis] - min[axis])
                .fold(0.0_f64, f64::max);
            (
                size,
                from_fn::<_, 3, _>(|axis| 0.5 * (min[axis] + max[axis])),
            )
        })
        .collect();
    let finest = cells
        .iter()
        .map(|&(size, _)| size)
        .fold(f64::INFINITY, f64::min);
    (
        finest,
        cells
            .iter()
            .filter(|&&(size, _)| size < finest * 1.5)
            .map(|&(_, center)| center)
            .collect(),
    )
}

#[test]
fn a_pore_refines_without_dragging_the_cube_down() {
    let radius = 0.1;
    let (finest, cells) = refinement(&pored_cube(radius), curvature(Quantity::new(1.0e-3)));
    assert!(finest < radius, "the pore is resolved at all");
    cells.iter().for_each(|center| {
        let distance = (0..3)
            .map(|axis| (center[axis] - 0.5).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            distance < 3.0 * radius,
            "a finest cell sits at {center:?}, away from the pore"
        )
    });
}

#[test]
fn a_tolerance_does_not_refine_a_polyhedron() {
    let (points, faces) = cube(4);
    let cube = tessellate(points, faces);
    let (without, _) = refinement(&cube, CurvatureSizing::default());
    let (tight, _) = refinement(&cube, curvature(Quantity::new(1.0e-4)));
    assert_eq!(
        tight, without,
        "no tolerance is tight enough to curve a flat face"
    );
}
