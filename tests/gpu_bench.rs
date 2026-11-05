#![cfg(feature = "gpu")]

//! Explicit dynamics on structured hexahedral grids, on the calling thread, on threads, and on a
//! CUDA device.
//!
//! The cost of a step is the difference in the time of two runs of different lengths divided by
//! the difference in their steps, which leaves out what a run does once: the masses, and on a
//! device the compiling of the kernels and the copying of the blocks. That once-only cost is
//! reported as the setup. A run on a device keeps its state there and keeps no history, so the
//! cost of a step is that of the forces and the update, whereas the integrator of the library
//! keeps the history of a run, so the runs on the calling thread and on threads are short and
//! those on the device long.
//!
//! Run alone, one test at a time, in release mode, so that only the library and this file are
//! compiled and nothing else competes for the cores or the device:
//! `cargo test --release -F gpu --test gpu_bench -- --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `CELLS` is the cells on a side of each grid, `45,90` by
//! default, `THREADS` is the threads, all those of the machine by default, `DEVICE` is the
//! ordinal of the device, and `REPETITIONS` is how many times each run is timed, three by
//! default.

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            DeviceExplicitDynamics, ElasticExplicitDynamics, Execution, GpuOptions, GpuPrecision,
            Lumped, Unchecked,
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use std::{thread::available_parallelism, time::Instant};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

const STEP: f64 = 1e-8;

fn variable<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn cells() -> Vec<usize> {
    std::env::var("CELLS")
        .unwrap_or_else(|_| "45,90".to_string())
        .split(',')
        .filter_map(|cells| cells.trim().parse().ok())
        .collect()
}

fn threads() -> usize {
    variable("THREADS").unwrap_or_else(|| available_parallelism().map_or(1, usize::from))
}

fn repetitions() -> usize {
    variable("REPETITIONS").unwrap_or(3)
}

fn lattice(cells: usize) -> Vec<[f64; 3]> {
    (0..=cells)
        .flat_map(|k| {
            (0..=cells).flat_map(move |j| {
                (0..=cells).map(move |i| {
                    [
                        i as f64 / cells as f64,
                        j as f64 / cells as f64,
                        k as f64 / cells as f64,
                    ]
                })
            })
        })
        .collect()
}

fn hexahedra(cells: usize) -> Vec<[usize; 8]> {
    let index = |i: usize, j: usize, k: usize| (k * (cells + 1) + j) * (cells + 1) + i;
    (0..cells)
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
        .collect()
}

fn grid(cells: usize) -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(cells);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    (
        (
            Grid::from((
                NeoHookean {
                    shear_modulus: Stress::pascals(3.0e9),
                    bulk_modulus: Stress::pascals(13.0e9),
                },
                Density::kilograms_per_cubic_meter(7.8e3),
                hexahedra(cells),
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

fn components<T, U>(nodes: &T) -> Vec<f64>
where
    T: Tensor,
    T::Item: Tensor<Item = Quantity<U>>,
{
    nodes
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_of(mut operation: impl FnMut()) -> f64 {
    let start = Instant::now();
    operation();
    start.elapsed().as_secs_f64() * 1e3
}

/// The cost of a step and of the setup of a run, in milliseconds, from runs of `short` and
/// `long` steps, each the median of the repetitions.
fn marginal(short: usize, long: usize, mut run: impl FnMut(usize)) -> (f64, f64) {
    let mut median_of =
        |steps| median((0..repetitions()).map(|_| time_of(|| run(steps))).collect());
    let (short_time, long_time) = (median_of(short), median_of(long));
    let step = (long_time - short_time) / (long - short) as f64;
    (step, short_time - short as f64 * step)
}

/// The steps of the runs that keep a history, which is large, so fewer for larger grids.
fn history_steps(nodes: usize) -> (usize, usize) {
    let short = (400_000 / nodes).clamp(1, 20);
    (short, 3 * short + 1)
}

/// Whether an error is that there is no device, which these tests skip.
fn no_device(error: &impl std::fmt::Debug) -> bool {
    format!("{error:?}").contains("No CUDA device is available")
}

fn options(precision: GpuPrecision) -> GpuOptions {
    GpuOptions {
        device: variable("DEVICE").unwrap_or(0),
        precision,
    }
}

/// Times the runs on a grid of `cells` cells on a side, and checks that the device gives the
/// state of the run on the calling thread.
fn compare(cells: usize) -> bool {
    let (model, reference) = grid(cells);
    let current = deformed(&reference);
    let nodes = current.len();
    let elements = cells.pow(3);
    let velocities = || NodalVelocities::from(vec![[0.0; 3]; nodes]);
    let conditions = BoundaryConditions::<Solid<3>>::new();
    let integrator = VelocityVerlet::new(Time::seconds(STEP));
    let history = |execution: Execution, steps: usize| {
        model
            .integrate(
                execution,
                Unchecked,
                Lumped,
                &integrator,
                &[Time::seconds(0.0), Time::seconds(steps as f64 * STEP)],
                (current.clone(), velocities()),
                &(),
                &conditions,
            )
            .unwrap()
    };
    let device = |precision: GpuPrecision, steps: usize| {
        model.integrate_on_device(
            options(precision),
            Time::seconds(STEP),
            steps,
            (current.clone(), velocities()),
            &conditions,
        )
    };
    // Whether there is a device, and whether it agrees with the calling thread.
    let (short, long) = history_steps(nodes);
    let expected = components(history(Execution::Serial, short).1.iter().last().unwrap());
    match device(GpuPrecision::F64, short) {
        Ok((found, _)) => {
            let found = components(&found);
            let gap = found
                .iter()
                .zip(&expected)
                .map(|(found, expected)| (found - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                gap < 1e-9,
                "the device differs from the calling thread by {gap}"
            );
        }
        Err(error) if no_device(&error) => {
            println!("skipped: {error:?}");
            return false;
        }
        Err(error) => panic!("{error:?}"),
    }
    let (serial, serial_setup) =
        marginal(short, long, |steps| drop(history(Execution::Serial, steps)));
    let (threaded, threaded_setup) = marginal(short, long, |steps| {
        drop(history(Execution::Threads(threads()), steps))
    });
    let (double, double_setup) = marginal(50, 550, |steps| {
        device(GpuPrecision::F64, steps).unwrap();
    });
    let (single, single_setup) = marginal(50, 550, |steps| {
        device(GpuPrecision::F32, steps).unwrap();
    });
    let rate = |step: f64| elements as f64 / step / 1e3;
    println!(
        "\n{elements} elements, {nodes} nodes ({cells}^3), {} threads",
        threads()
    );
    println!(
        "{:>16} {:>11} {:>14} {:>10} {:>12}",
        "", "ms/step", "Melement/s", "vs threads", "setup ms"
    );
    for (label, step, setup) in [
        ("calling thread", serial, serial_setup),
        ("threads", threaded, threaded_setup),
        ("device f64", double, double_setup),
        ("device f32", single, single_setup),
    ] {
        println!(
            "{label:>16} {step:>11.3} {:>14.1} {:>9.2}x {setup:>12.1}",
            rate(step),
            threaded / step
        );
    }
    true
}

#[test]
#[ignore]
fn single_block_steps() {
    for cells in cells() {
        if !compare(cells) {
            return;
        }
    }
}
