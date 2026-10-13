use super::{CornerEnergy, Incidence, element_energy};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, Verdict},
    },
    math::{Scalar, Tensor},
};

fn grid(n: usize) -> (Vec<[usize; 8]>, Vec<[f64; 3]>) {
    let id = |i: usize, j: usize, k: usize| i + (n + 1) * (j + (n + 1) * k);
    let mut coordinates = Vec::new();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                coordinates.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    let mut connectivity = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                connectivity.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    (connectivity, coordinates)
}

fn mesh(connectivity: Vec<[usize; 8]>, coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Hexahedral(connectivity.into())],
        coordinates.into(),
    ))
}

fn minimum_scaled(mesh: &Mesh<3>) -> f64 {
    mesh.minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn reflected_element_is_invisible_to_cosine_energy() {
    let (connectivity, coordinates) = grid(1);
    let element = connectivity[0];
    let reflected = coordinates.iter().map(|&[x, y, z]| [x, y, -z]).collect();
    let perfect = mesh(connectivity.clone(), coordinates);
    let reflected = mesh(connectivity, reflected);
    let energy = |mesh: &Mesh<3>, energy| element_energy(&element, mesh.coordinates(), energy);
    assert!(energy(&perfect, CornerEnergy::Signed).abs() < 1.0e-12);
    assert!(energy(&perfect, CornerEnergy::Cosine).abs() < 1.0e-12);
    assert!((energy(&reflected, CornerEnergy::Signed) - 48.0).abs() < 1.0e-12);
    assert!(energy(&reflected, CornerEnergy::Cosine).abs() < 1.0e-12);
}

#[test]
fn perfect_grid_does_not_move() {
    let (connectivity, coordinates) = grid(2);
    let mut mesh = mesh(connectivity, coordinates);
    let before = mesh.coordinates().iter().cloned().collect::<Vec<_>>();
    mesh.corner_energy_smooth(5, CornerEnergy::Signed);
    mesh.coordinates()
        .iter()
        .zip(before)
        .for_each(|(now, was)| assert!((now - &was).norm().value() < 1.0e-12));
}

#[test]
fn signed_energy_untangles_an_inverted_interior_node() {
    let (connectivity, mut coordinates) = grid(2);
    coordinates[13] = [1.0, 1.0, 2.5];
    let mut mesh = mesh(connectivity, coordinates);
    assert!(minimum_scaled(&mesh) < 0.0);
    mesh.corner_energy_smooth(10, CornerEnergy::Signed);
    assert!(minimum_scaled(&mesh) > 0.0);
}

const SWEEPS: usize = 20;
const SEEDS: u64 = 5;

#[derive(Clone, Copy)]
enum Arm {
    None,
    Laplace,
    SmartLaplace,
    Energy(CornerEnergy),
}

impl Arm {
    fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Laplace => "laplace",
            Self::SmartLaplace => "smart_laplace",
            Self::Energy(CornerEnergy::Cosine) => "cosine",
            Self::Energy(CornerEnergy::Signed) => "signed",
        }
    }
}

const ARMS: [Arm; 5] = [
    Arm::None,
    Arm::Laplace,
    Arm::SmartLaplace,
    Arm::Energy(CornerEnergy::Cosine),
    Arm::Energy(CornerEnergy::Signed),
];

fn ball(n: usize) -> (Vec<[usize; 8]>, Vec<[f64; 3]>, Vec<bool>) {
    let (connectivity, coordinates) = grid(n);
    let half = n as f64 / 2.0;
    let mut fixed = Vec::with_capacity(coordinates.len());
    let coordinates = coordinates
        .into_iter()
        .map(|point| {
            let p = point.map(|c| (c - half) / half);
            let infinity = p.iter().fold(0.0, |m: f64, c| m.max(c.abs()));
            let euclid = p.iter().map(|c| c * c).sum::<f64>().sqrt();
            fixed.push(infinity == 1.0);
            if euclid > 0.0 {
                p.map(|c| c * infinity / euclid)
            } else {
                p
            }
        })
        .collect();
    (connectivity, coordinates, fixed)
}

fn jitter(coordinates: &mut [[f64; 3]], fixed: &[bool], amplitude: f64, seed: u64) {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut uniform = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    };
    coordinates
        .iter_mut()
        .zip(fixed)
        .filter(|(_, fixed)| !**fixed)
        .for_each(|(point, _)| point.iter_mut().for_each(|c| *c += amplitude * uniform()));
}

fn laplace(mesh: &mut Mesh<3>, fixed: &[bool], smart: bool) {
    let neighbors = mesh.node_node_connectivity().to_vec();
    let incidence = Incidence::of(mesh);
    let coordinates = mesh.coordinates.members_mut();
    for _ in 0..SWEEPS {
        for node in 0..neighbors.len() {
            if fixed[node] {
                continue;
            }
            let before = incidence.minimum_scaled_jacobian(node, coordinates);
            let original = coordinates[node].clone();
            coordinates[node] = neighbors[node]
                .iter()
                .map(|&neighbor| &coordinates[neighbor])
                .sum::<Coordinate<3>>()
                / (neighbors[node].len() as Scalar);
            if smart && incidence.minimum_scaled_jacobian(node, coordinates) < before {
                coordinates[node] = original;
            }
        }
    }
}

fn run(arm: Arm, mesh: &mut Mesh<3>, fixed: &[bool]) {
    match arm {
        Arm::None => {}
        Arm::Laplace => laplace(mesh, fixed, false),
        Arm::SmartLaplace => laplace(mesh, fixed, true),
        Arm::Energy(energy) => mesh.corner_energy_smooth_fixed(SWEEPS, energy, fixed),
    }
}

struct Summary {
    minimum: f64,
    p10: f64,
    median: f64,
    inverted: usize,
    edge_ratio: f64,
    bins: [usize; 6],
}

fn summarize(mesh: &Mesh<3>) -> (Summary, Vec<f64>) {
    let mut msj = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let raw = msj.clone();
    msj.sort_by(f64::total_cmp);
    let quantile = |q: f64| msj[((msj.len() - 1) as f64 * q).round() as usize];
    let mut bins = [0; 6];
    msj.iter().for_each(|&value| {
        let bin = if value <= 0.0 {
            0
        } else {
            1 + ((value * 5.0).ceil() as usize - 1).min(4)
        };
        bins[bin] += 1;
    });
    let edge_ratio = mesh
        .maximum_edge_ratios()
        .into_iter()
        .flatten()
        .fold(0.0, f64::max);
    (
        Summary {
            minimum: msj[0],
            p10: quantile(0.1),
            median: quantile(0.5),
            inverted: msj.iter().filter(|&&value| value <= 0.0).count(),
            edge_ratio,
            bins,
        },
        raw,
    )
}

fn report(model: &str, seed: u64, arm: Arm, summary: &Summary, seconds: f64) {
    let [b0, b1, b2, b3, b4, b5] = summary.bins;
    println!(
        "{model},{seed},{},{:.4},{:.4},{:.4},{},{:.3},{b0},{b1},{b2},{b3},{b4},{b5},{seconds:.2}",
        arm.name(),
        summary.minimum,
        summary.p10,
        summary.median,
        summary.inverted,
        summary.edge_ratio,
    );
}

fn experiment(model: &str, n: usize, amplitude: f64, seeds: u64, output: Option<&str>) {
    for seed in 0..seeds {
        let (connectivity, mut coordinates, fixed) = ball(n);
        if amplitude > 0.0 {
            jitter(&mut coordinates, &fixed, amplitude * 2.0 / n as f64, seed);
        }
        for arm in ARMS {
            let mut mesh = mesh(connectivity.clone(), coordinates.clone());
            let start = std::time::Instant::now();
            run(arm, &mut mesh, &fixed);
            let seconds = start.elapsed().as_secs_f64();
            let moved = mesh
                .coordinates()
                .iter()
                .zip(&coordinates)
                .zip(&fixed)
                .filter(|(_, fixed)| **fixed)
                .any(|((now, was), _)| (now - &Coordinate::from(*was)).norm().value() > 0.0);
            assert!(!moved, "a fixed node moved");
            let (summary, raw) = summarize(&mesh);
            report(model, seed, arm, &summary, seconds);
            if let Some(directory) = output {
                let path = format!("{directory}/{model}_seed{seed}_{}.csv", arm.name());
                let text = raw
                    .iter()
                    .map(|value| format!("{value}\n"))
                    .collect::<String>();
                std::fs::write(path, text).unwrap();
            }
        }
    }
}

#[test]
#[ignore]
fn corner_energy_experiment() {
    let output = std::env::var("CORNER_ENERGY_OUT").ok();
    if let Some(directory) = &output {
        std::fs::create_dir_all(directory).unwrap();
    }
    let output = output.as_deref();
    println!(
        "model,seed,arm,min_msj,p10_msj,median_msj,inverted,max_edge_ratio,\
         le0,0-0.2,0.2-0.4,0.4-0.6,0.6-0.8,0.8-1,seconds"
    );
    for n in [8, 16] {
        experiment(&format!("m1_n{n}"), n, 0.0, 1, output);
    }
    for n in [8, 16] {
        for amplitude in [0.3, 0.5, 0.7] {
            experiment(
                &format!("m2_n{n}_a{amplitude}"),
                n,
                amplitude,
                SEEDS,
                output,
            );
        }
    }
}
