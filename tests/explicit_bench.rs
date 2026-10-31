#![cfg(feature = "fem")]

//! Serial baseline for explicit dynamics on structured hexahedral grids.
//!
//! Run alone, so only the library and this file are compiled:
//! `CARGO_BUILD_JOBS=1 cargo test --release -F fem --test explicit_bench -- --ignored --nocapture`

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            ElasticExplicitDynamics,
            elastic::{ElasticElements, SEQUENTIAL_BLOCKS},
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use std::{hint::black_box, ops::Range, sync::atomic::Ordering, time::Instant};

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

type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Three = Blocks<Grid, Blocks<Steel, Grid>>;

const COLUMNS: usize = 32;

/// Hexahedra filling the layers `layers` of a `COLUMNS` by `COLUMNS` column grid.
fn slab(layers: Range<usize>) -> Vec<[usize; 8]> {
    let nodes = COLUMNS + 1;
    let index = |i: usize, j: usize, k: usize| (k * nodes + j) * nodes + i;
    layers
        .flat_map(|k| {
            (0..COLUMNS).flat_map(move |j| {
                (0..COLUMNS).map(move |i| {
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
        .collect()
}

/// Three slabs of about 100k, 30k and 1k elements, with different constitutive models, that share
/// the nodes on their interfaces.
fn three_blocks() -> (Model<Three, 3>, Vec<[f64; 3]>) {
    let (first, second) = (98, 29);
    let layers = first + second + 1;
    let nodes = COLUMNS + 1;
    let coordinates = (0..=layers)
        .flat_map(|k| {
            (0..nodes).flat_map(move |j| {
                (0..nodes).map(move |i| {
                    [
                        i as f64 / COLUMNS as f64,
                        j as f64 / COLUMNS as f64,
                        k as f64 / COLUMNS as f64,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let density = Density::kilograms_per_cubic_meter(7.8e3);
    let rubber = |shear: f64, bulk: f64| NeoHookean {
        shear_modulus: Stress::pascals(shear),
        bulk_modulus: Stress::pascals(bulk),
    };
    let blocks = Blocks(
        Grid::from((rubber(3.0e9, 13.0e9), density, slab(0..first), &reference)),
        Blocks(
            Steel::from((
                SaintVenantKirchhoff {
                    bulk_modulus: Stress::pascals(160.0e9),
                    shear_modulus: Stress::pascals(80.0e9),
                },
                density,
                slab(first..first + second),
                &reference,
            )),
            Grid::from((
                rubber(1.0e9, 5.0e9),
                density,
                slab(first + second..layers),
                &reference,
            )),
        ),
    );
    ((blocks, reference).into(), coordinates)
}

fn checksum(
    forces: &impl Tensor<Item = impl Tensor<Item = Quantity<conspire::units::Force>>>,
) -> u64 {
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value().to_bits()))
        .fold(0u64, |sum, bits| sum.rotate_left(5) ^ bits)
}

#[test]
#[ignore]
fn heterogeneous_blocks() {
    let (model, coordinates) = three_blocks();
    let current = deformed(&coordinates);
    println!(
        "\nthree blocks, {} nodes: 100,352 + 29,696 + 1,024 elements",
        coordinates.len()
    );
    let mut results = Vec::new();
    for sequential in [true, false, true, false] {
        SEQUENTIAL_BLOCKS.store(sequential, Ordering::Relaxed);
        let samples = (0..30)
            .map(|_| {
                let start = Instant::now();
                black_box(model.nodal_forces(black_box(&current)).unwrap());
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect::<Vec<_>>();
        let sum = checksum(&model.nodal_forces(&current).unwrap());
        let time = median(samples);
        println!(
            "{:>22}: {:>8.3} ms   checksum {:016x}",
            if sequential {
                "blocks one at a time"
            } else {
                "blocks joined"
            },
            time,
            sum
        );
        results.push(sum);
    }
    assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
}

type Four = Blocks<Grid, Blocks<Grid, Blocks<Grid, Grid>>>;

/// Four small slabs of 1, 2, 3 and 4 layers, about 1k to 4k elements each.
fn four_blocks() -> (Model<Four, 3>, Vec<[f64; 3]>) {
    let layers = 10;
    let nodes = COLUMNS + 1;
    let coordinates = (0..=layers)
        .flat_map(|k| {
            (0..nodes).flat_map(move |j| {
                (0..nodes).map(move |i| {
                    [
                        i as f64 / COLUMNS as f64,
                        j as f64 / COLUMNS as f64,
                        k as f64 / COLUMNS as f64,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let density = Density::kilograms_per_cubic_meter(7.8e3);
    let block = |shear: f64, layers: Range<usize>| {
        Grid::from((
            NeoHookean {
                shear_modulus: Stress::pascals(shear),
                bulk_modulus: Stress::pascals(4.0 * shear),
            },
            density,
            slab(layers),
            &reference,
        ))
    };
    let blocks = Blocks(
        block(1.0e9, 0..1),
        Blocks(
            block(2.0e9, 1..3),
            Blocks(block(3.0e9, 3..6), block(4.0e9, 6..10)),
        ),
    );
    ((blocks, reference).into(), coordinates)
}

#[test]
#[ignore]
fn small_blocks() {
    let (model, coordinates) = four_blocks();
    let current = deformed(&coordinates);
    println!(
        "\nfour blocks, {} nodes: 1,024 + 2,048 + 3,072 + 4,096 elements",
        coordinates.len()
    );
    let mut results = Vec::new();
    for sequential in [true, false, true, false] {
        SEQUENTIAL_BLOCKS.store(sequential, Ordering::Relaxed);
        let samples = (0..300)
            .map(|_| {
                let start = Instant::now();
                black_box(model.nodal_forces(black_box(&current)).unwrap());
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect::<Vec<_>>();
        let sum = checksum(&model.nodal_forces(&current).unwrap());
        println!(
            "{:>22}: {:>8.3} ms   checksum {:016x}",
            if sequential {
                "blocks one at a time"
            } else {
                "blocks joined"
            },
            median(samples),
            sum
        );
        results.push(sum);
    }
    assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
}
