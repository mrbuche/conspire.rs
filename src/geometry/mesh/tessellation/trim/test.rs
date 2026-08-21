use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Fitting, Mesh, Verdict,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::Scalar,
};
use std::f64::consts::TAU;

fn torus(major: Scalar, minor: Scalar, around: usize, tube: usize) -> Tessellation {
    let mut coordinates = Vec::new();
    (0..around).for_each(|i| {
        let theta = TAU * i as Scalar / around as Scalar;
        (0..tube).for_each(|j| {
            let phi = TAU * j as Scalar / tube as Scalar;
            let radius = major + minor * phi.cos();
            coordinates.push([
                radius * theta.cos(),
                radius * theta.sin(),
                minor * phi.sin(),
            ])
        })
    });
    let index = |i: usize, j: usize| (i % around) * tube + (j % tube);
    let faces: Vec<[usize; D]> = (0..around)
        .flat_map(|i| {
            (0..tube).flat_map(move |j| {
                [
                    [index(i, j), index(i + 1, j), index(i + 1, j + 1)],
                    [index(i, j), index(i + 1, j + 1), index(i, j + 1)],
                ]
            })
        })
        .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

#[test]
fn dualized_slender_torus_is_not_inverted() {
    let tessellation = torus(1.0, 0.15, 64, 24);
    let mut octree =
        Octree::<u16, usize>::from_features(&tessellation, 3.0, CurvatureSizing::default(), 0)
            .unwrap();
    octree
        .equilibrate(Balancing::Strong(1), Pairing::Regular)
        .unwrap();
    let mut mesh = octree.dualize();
    let background = mesh.number_of_elements();
    tessellation.trim(&mut mesh).unwrap();
    assert!(mesh.number_of_elements() < background);
    let mesh = mesh.buffer(&tessellation, Fitting::Snap).unwrap();
    let worst = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min);
    assert!(worst > 0.15, "{worst}");
}

#[test]
fn trim_keeps_the_same_cells_at_any_scale() {
    let trimmed = |scale: Scalar| {
        let tessellation = torus(scale, 0.15 * scale, 64, 24);
        let mut octree =
            Octree::<u16, usize>::from_features(&tessellation, 3.0, CurvatureSizing::default(), 0)
                .unwrap();
        octree
            .equilibrate(Balancing::Strong(1), Pairing::Regular)
            .unwrap();
        let mut mesh = octree.dualize();
        let background = mesh.number_of_elements();
        tessellation.trim(&mut mesh).unwrap();
        (background, mesh.number_of_elements())
    };
    let (background, kept) = trimmed(1.0);
    assert_eq!(trimmed(1.0e-4), (background, kept));
}
