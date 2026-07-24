use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
        ntree::{CurvatureSizing, Octree},
    },
    math::Scalar,
};
use std::f64::consts::{PI, TAU};

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

fn curvature(tolerance: Scalar) -> CurvatureSizing {
    CurvatureSizing {
        tolerance: Some(tolerance),
        ..Default::default()
    }
}

#[test]
fn tighter_curvature_tolerance_refines_more() {
    let tessellation = sphere(4, 8, 2.0);
    let scale = 4.0;
    let loose = Octree::<u16, usize>::from_features(&tessellation, scale, curvature(1.0), 0);
    let medium = Octree::<u16, usize>::from_features(&tessellation, scale, curvature(1.0e-2), 0);
    let tight = Octree::<u16, usize>::from_features(&tessellation, scale, curvature(1.0e-3), 0);
    assert!(medium.len() > loose.len());
    assert!(tight.len() > medium.len());
}

#[test]
fn default_curvature_sizing_disables_curvature_refinement() {
    let tessellation = sphere(4, 8, 2.0);
    let scale = 4.0;
    let without = Octree::<u16, usize>::from_features(&tessellation, scale, curvature(1.0e-3), 0);
    let with_default =
        Octree::<u16, usize>::from_features(&tessellation, scale, CurvatureSizing::default(), 0);
    assert!(with_default.len() <= without.len());
}
