use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Fitting, Mesh, Tessellation, Verdict},
    },
    math::{Quantity, Scalar},
};
use std::f64::consts::TAU;

const THRESHOLD: Scalar = 0.1;

fn oblique_ridge(angle: Scalar) -> Tessellation {
    let (s, c) = angle.sin_cos();
    let coordinates = Coordinates::from(
        [
            [0.0, -1.5, 0.0],
            [2.0, -1.5, 0.0],
            [2.0, 1.5, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
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
            let a = TAU * i as f64 / segments as f64;
            [radius * a.cos(), radius * a.sin(), 0.0]
        })
        .chain((0..segments).map(|i| {
            let a = TAU * i as f64 / segments as f64;
            [radius * a.cos(), radius * a.sin(), height]
        }))
        .collect();
    points.push([0.0, 0.0, 0.0]);
    points.push([0.0, 0.0, height]);
    let (bottom, top) = (2 * segments, 2 * segments + 1);
    let mut triangles = Vec::<[usize; 3]>::new();
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
        .buffer_targeted(&target, Fitting::Soft, THRESHOLD)
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
    let targeted = background(&target, 0.35)
        .buffer_targeted(&target, Fitting::Snap, THRESHOLD)
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
        fans * 10 < targeted.number_of_elements(),
        "{fans} pyramids in {} cells is not targeted",
        targeted.number_of_elements()
    );
    let [_, Connectivity::Pyramidal(_)] = targeted.connectivities() else {
        unreachable!("one hexahedral block and one pyramidal block")
    };
}

#[test]
fn buffer_targeted_is_never_worse_than_buffer() {
    let target = oblique_ridge(40.0_f64.to_radians());
    let plain = background(&target, 0.35)
        .buffer(&target, Fitting::Snap)
        .unwrap();
    let targeted = background(&target, 0.35)
        .buffer_targeted(&target, Fitting::Snap, 0.2)
        .unwrap();
    assert!(
        worst(&targeted) >= worst(&plain),
        "targeted {} vs plain {}",
        worst(&targeted),
        worst(&plain)
    );
}

fn tally(label: &str, mesh: &Mesh<3>, seconds: f64) {
    let all: Vec<Scalar> = mesh
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .copied()
        .collect();
    let below = |t: Scalar| all.iter().filter(|&&q| q < t).count();
    let count = |kind: fn(&Connectivity) -> bool| {
        mesh.connectivities()
            .iter()
            .filter(|c| kind(c))
            .flatten()
            .count()
    };
    let per_kind: Vec<String> = mesh
        .connectivities()
        .iter()
        .zip(mesh.minimum_scaled_jacobians())
        .map(|(c, q)| {
            let name = match c {
                Connectivity::Hexahedral(_) => "hex",
                Connectivity::Pyramidal(_) => "pyr",
                Connectivity::Tetrahedral(_) => "tet",
                _ => "?",
            };
            let min = q.iter().copied().fold(Scalar::INFINITY, Scalar::min);
            let median = {
                let mut sorted = q.clone();
                sorted.sort_by(Scalar::total_cmp);
                sorted[sorted.len() / 2]
            };
            format!("{name} min {min:.3} med {median:.3}")
        })
        .collect();
    eprintln!("          [{}]", per_kind.join(" | "));
    eprintln!(
        "  {label:>6}: cells {:>5} pyr {:>4} tet {:>4} worst {:>6.3} <0.1: {:>3} <0.2: {:>3} <0.3: {:>3} ({seconds:.1}s)",
        all.len(),
        count(|c| matches!(c, Connectivity::Pyramidal(_))),
        count(|c| matches!(c, Connectivity::Tetrahedral(_))),
        worst(mesh),
        below(0.1),
        below(0.2),
        below(0.3),
    );
}

#[test]
fn tmp_compare_templates() {
    let cases: Vec<(String, Tessellation, Scalar)> = vec![
        ("cyl.35".into(), cylinder(1.5, 2.0, 32), 0.35),
        ("cyl.3".into(), cylinder(1.5, 2.0, 32), 0.3),
        ("rid0".into(), oblique_ridge(0.0), 0.35),
        ("rid20".into(), oblique_ridge(20.0_f64.to_radians()), 0.35),
        ("rid40".into(), oblique_ridge(40.0_f64.to_radians()), 0.35),
    ];
    for (name, target, size) in cases {
        for threshold in [0.1, 0.2] {
            eprintln!("{name} threshold {threshold}");
            let start = std::time::Instant::now();
            let plain = background(&target, size)
                .buffer(&target, Fitting::Snap)
                .unwrap();
            tally("hex", &plain, start.elapsed().as_secs_f64());
            for (label, template) in [
                ("pyr", super::Template::Pyramids),
                ("pyr+tet", super::Template::PyramidsAndTets),
            ] {
                let start = std::time::Instant::now();
                let mesh = background(&target, size)
                    .targeted(&target, Fitting::Snap, threshold, template)
                    .unwrap();
                tally(label, &mesh, start.elapsed().as_secs_f64());
            }
        }
    }
}
