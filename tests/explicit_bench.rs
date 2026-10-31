#![cfg(feature = "fem")]

//! Serial baseline for explicit dynamics on structured hexahedral grids.
//!
//! Run alone, so only the library and this file are compiled:
//! `CARGO_BUILD_JOBS=1 cargo test --release -F fem --test explicit_bench -- --ignored --nocapture`

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{ElasticExplicitDynamics, elastic::ElasticElements},
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use std::{hint::black_box, time::Instant};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

const STEPS: usize = 50;

fn model(cells: usize) -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let nodes = cells + 1;
    let index = |i: usize, j: usize, k: usize| (k * nodes + j) * nodes + i;
    let spacing = 1.0 / cells as f64;
    let coordinates = (0..nodes)
        .flat_map(|k| {
            (0..nodes).flat_map(move |j| {
                (0..nodes)
                    .map(move |i| [i as f64 * spacing, j as f64 * spacing, k as f64 * spacing])
            })
        })
        .collect::<Vec<_>>();
    let connectivity = (0..cells)
        .flat_map(|k| {
            (0..cells).flat_map(move |j| {
                (0..cells).map(move |i| {
                    [
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ]
                })
            })
        })
        .collect::<Vec<[usize; 8]>>();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    (
        (
            Grid::from((
                NeoHookean {
                    shear_modulus: Stress::pascals(3.0e9),
                    bulk_modulus: Stress::pascals(13.0e9),
                },
                Density::kilograms_per_cubic_meter(7.8e3),
                connectivity,
                &reference,
            )),
            reference,
        )
            .into(),
        coordinates,
    )
}

/// A smooth, small displacement so that every element has a nonzero stress.
fn deformed(coordinates: &[[f64; 3]]) -> NodalCoordinates<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                x + 1e-3 * (3.0 * y).sin(),
                y + 1e-3 * (2.0 * z).sin(),
                z + 1e-3 * (4.0 * x).sin(),
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

#[test]
#[ignore]
fn explicit_baseline() {
    println!(
        "\n{:>9} {:>9} {:>14} {:>14} {:>14}",
        "elements", "nodes", "forces (ms)", "step (ms)", "forces/step"
    );
    for cells in [10, 22, 46] {
        let (model, coordinates) = model(cells);
        let current = deformed(&coordinates);
        let repetitions = (2_000_000 / cells.pow(3)).clamp(5, 200);
        let forces = median(
            (0..repetitions)
                .map(|_| {
                    let start = Instant::now();
                    black_box(model.nodal_forces(black_box(&current)).unwrap());
                    start.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        let checksum = model
            .nodal_forces(&current)
            .unwrap()
            .iter()
            .flat_map(|force| force.iter().map(|component| component.value().to_bits()))
            .fold(0u64, |sum, bits| sum.rotate_left(5) ^ bits);
        println!("checksum {cells}: {checksum:016x}");
        let masses = model.nodal_lumped_masses();
        let dt = 1e-8;
        let time = [Time::seconds(0.0), Time::seconds(STEPS as f64 * dt)];
        let velocities = NodalVelocities::from(vec![[0.0; 3]; coordinates.len()]);
        let start = Instant::now();
        black_box(
            model
                .integrate(
                    &VelocityVerlet::new(Time::seconds(dt)),
                    &time,
                    (current.clone(), velocities),
                    &(),
                    &masses,
                    &BoundaryConditions::<Solid<3>>::new(),
                )
                .unwrap(),
        );
        let step = start.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
        println!(
            "{:>9} {:>9} {:>14.3} {:>14.3} {:>13.0}%",
            cells.pow(3),
            (cells + 1).pow(3),
            forces,
            step,
            100.0 * forces / step
        );
    }
}
