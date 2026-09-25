use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Fitting, Mesh, Tessellation, Verdict},
    },
    math::{Quantity, Scalar},
};

fn oblique_ridge(angle: Scalar) -> Tessellation {
    let (s, c) = angle.sin_cos();
    let coordinates = Coordinates::from(
        [
            [0.0, -1.5, 0.0],
            [4.0, -1.5, 0.0],
            [4.0, 1.5, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
        ]
        .map(|[x, y, z]| [c * x - s * y, s * x + c * y, z])
        .to_vec(),
    );
    let triangles: Vec<[usize; 3]> = vec![
        [0, 2, 1],
        [0, 3, 2],
        [2, 3, 4],
        [2, 4, 5],
        [0, 1, 4],
        [1, 5, 4],
        [0, 4, 3],
        [1, 2, 5],
    ];
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    )))
}

fn worst(mesh: &Mesh<3>) -> Scalar {
    mesh.minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality))
}

fn pyramids(mesh: &Mesh<3>) -> usize {
    mesh.connectivities()
        .iter()
        .filter(|connectivity| matches!(connectivity, Connectivity::Pyramidal(_)))
        .flatten()
        .count()
}

fn background(target: &Tessellation, size: Scalar) -> Mesh<3> {
    let (mut background, _) = target.lattice_background(Quantity::new(size)).unwrap();
    target.trim(&mut background).unwrap();
    background
}

fn cylinder(radius: Scalar, height: Scalar, segments: usize) -> Tessellation {
    let mut points: Vec<[Scalar; 3]> = (0..segments)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / segments as f64;
            [radius * a.cos(), radius * a.sin(), 0.0]
        })
        .chain((0..segments).map(|i| {
            let a = std::f64::consts::TAU * i as f64 / segments as f64;
            [radius * a.cos(), radius * a.sin(), height]
        }))
        .collect();
    points.push([0.0, 0.0, 0.0]);
    points.push([0.0, 0.0, height]);
    let (bottom, top) = (2 * segments, 2 * segments + 1);
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    for i in 0..segments {
        let j = (i + 1) % segments;
        triangles.push([i, j, segments + j]);
        triangles.push([i, segments + j, segments + i]);
        triangles.push([bottom, j, i]);
        triangles.push([top, segments + i, segments + j]);
    }
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        Coordinates::from(points),
    )))
}

#[test]
fn buffer_targeted_is_the_plain_buffer_under_soft_fitting() {
    let target = cylinder(1.5, 2.0, 32);
    let plain = background(&target, 0.35)
        .buffer(&target, Fitting::Soft)
        .unwrap();
    let targeted = background(&target, 0.35)
        .buffer_targeted(&target, Fitting::Soft)
        .unwrap();
    assert_eq!(pyramids(&targeted), 0);
    assert_eq!(targeted.number_of_elements(), plain.number_of_elements());
    assert_eq!(worst(&targeted), worst(&plain));
}

#[test]
fn buffer_targeted_fans_only_the_cells_snapping_ruined_on_a_cylinder() {
    let target = cylinder(1.5, 2.0, 32);
    let plain = background(&target, 0.35)
        .buffer(&target, Fitting::Snap)
        .unwrap();
    let mixed = background(&target, 0.35)
        .buffer_mixed(&target, Fitting::Snap)
        .unwrap();
    let targeted = background(&target, 0.35)
        .buffer_targeted(&target, Fitting::Snap)
        .unwrap();
    assert!(
        worst(&plain) < 0.1,
        "fixture no longer bowties: {}",
        worst(&plain)
    );
    assert!(
        worst(&targeted) > 0.1 && worst(&targeted) > 2.0 * worst(&plain),
        "targeted {} vs plain {}",
        worst(&targeted),
        worst(&plain)
    );
    let fans = pyramids(&targeted);
    assert!(fans > 0 && fans.is_multiple_of(5), "whole fans, got {fans}");
    assert!(
        fans * 4 < pyramids(&mixed),
        "{fans} vs {}",
        pyramids(&mixed)
    );
    let [_, Connectivity::Pyramidal(_)] = targeted.connectivities() else {
        unreachable!("one hexahedral block and one pyramidal block")
    };
}

#[test]
fn buffer_targeted_is_never_worse_than_buffer() {
    for degrees in [0.0_f64, 15.0, 30.0, 45.0] {
        for threshold in [0.1, 0.15, 0.25] {
            let target = oblique_ridge(degrees.to_radians());
            let plain = background(&target, 0.35)
                .buffer(&target, Fitting::Snap)
                .unwrap();
            let targeted = background(&target, 0.35)
                .targeted(&target, Fitting::Snap, threshold)
                .unwrap();
            assert!(
                worst(&targeted) >= worst(&plain),
                "{degrees} deg, threshold {threshold}: targeted {} vs plain {}",
                worst(&targeted),
                worst(&plain)
            );
        }
    }
}
