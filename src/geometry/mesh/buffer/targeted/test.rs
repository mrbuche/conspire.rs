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

fn report(label: &str, mesh: &Mesh<3>) {
    let all: Vec<Scalar> = mesh
        .minimum_scaled_jacobians()
        .iter()
        .flatten()
        .copied()
        .collect();
    let worst = all.iter().copied().fold(Scalar::INFINITY, Scalar::min);
    let below = |t: Scalar| all.iter().filter(|&&q| q < t).count();
    let pyramids = mesh
        .connectivities()
        .iter()
        .filter(|c| matches!(c, Connectivity::Pyramidal(_)))
        .flatten()
        .count();
    eprintln!(
        "{label:>9}: cells {:>5} pyramids {pyramids:>4} worst {worst:.3} <0.1: {} <0.2: {} <0.3: {}",
        all.len(),
        below(0.1),
        below(0.2),
        below(0.3),
    );
}

#[test]
fn diagnostic_oblique_ridge() {
    for degrees in [0.0_f64, 15.0, 30.0, 45.0] {
        let target = oblique_ridge(degrees.to_radians());
        let background = || {
            let (mut background, _) = target.lattice_background(Quantity::new(0.35)).unwrap();
            target.trim(&mut background).unwrap();
            background
        };
        eprintln!(
            "--- {degrees} deg, core hexes {}",
            background().number_of_elements()
        );
        for fitting in [Fitting::Soft, Fitting::Snap] {
            eprintln!("{fitting:?}");
            report("buffer", &background().buffer(&target, fitting).unwrap());
            report(
                "mixed",
                &background().buffer_mixed(&target, fitting).unwrap(),
            );
            report(
                "targeted",
                &background().buffer_targeted(&target, fitting).unwrap(),
            );
        }
    }
}
