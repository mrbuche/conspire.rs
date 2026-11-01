#![cfg(feature = "fem")]

//! Serial and parallel explicit dynamics on structured hexahedral grids.
//!
//! Run alone, one test at a time, so only the library and this file are compiled and the tests
//! do not compete for cores:
//! `CARGO_BUILD_JOBS=1 cargo test --release -F fem --test explicit_bench -- --ignored --nocapture --test-threads=1`
//! The threads default to 4 and can be set with `THREADS`.

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            ElasticExplicitDynamics, Execution, NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
            mass::InverseMass,
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use std::{hint::black_box, ops::Range, time::Instant};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

const STEPS: usize = 50;

fn threads() -> usize {
    std::env::var("THREADS")
        .ok()
        .and_then(|threads| threads.parse().ok())
        .unwrap_or(4)
}

fn density() -> Quantity<Density> {
    Density::kilograms_per_cubic_meter(7.8e3)
}

fn rubber(shear: f64, bulk: f64) -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(shear),
        bulk_modulus: Stress::pascals(bulk),
    }
}

/// Hexahedra filling the layers `layers` of an `nx` by `ny` column grid.
fn slab(nx: usize, ny: usize, layers: Range<usize>) -> Vec<[usize; 8]> {
    let index = |i: usize, j: usize, k: usize| (k * (ny + 1) + j) * (nx + 1) + i;
    layers
        .flat_map(|k| {
            (0..ny).flat_map(move |j| {
                (0..nx).map(move |i| {
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

fn lattice(nx: usize, ny: usize, layers: usize) -> Vec<[f64; 3]> {
    (0..=layers)
        .flat_map(|k| {
            (0..=ny).flat_map(move |j| {
                (0..=nx).map(move |i| {
                    [
                        i as f64 / nx as f64,
                        j as f64 / ny as f64,
                        k as f64 / nx as f64,
                    ]
                })
            })
        })
        .collect()
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

fn checksum<T, U>(forces: &T) -> u64
where
    T: Tensor,
    T::Item: Tensor<Item = Quantity<U>>,
{
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value().to_bits()))
        .fold(0u64, |sum, bits| sum.rotate_left(5) ^ bits)
}

fn time_of(repetitions: usize, mut operation: impl FnMut()) -> f64 {
    median(
        (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                operation();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

/// Times the serial and parallel nodal forces of a model, and checks that they are identical.
fn compare<B>(label: &str, model: &Model<B, 3>, current: &NodalCoordinates<3>, repetitions: usize)
where
    B: ElasticElements<3> + ParallelElasticElements<3>,
{
    let workers = Workers::new(threads()).unwrap();
    let mut scratch = model.scratch();
    let serial = time_of(repetitions, || {
        black_box(model.nodal_forces(black_box(current)).unwrap());
    });
    let parallel = time_of(repetitions, || {
        let mut forces = NodalForcesSolid::<3>::zero(current.len());
        model
            .nodal_forces_parallel_into(&mut scratch, &workers, black_box(current), &mut forces)
            .unwrap();
        black_box(forces);
    });
    let expected = checksum(&model.nodal_forces(current).unwrap());
    let mut forces = NodalForcesSolid::<3>::zero(current.len());
    model
        .nodal_forces_parallel_into(&mut scratch, &workers, current, &mut forces)
        .unwrap();
    let found = checksum(&forces);
    println!(
        "{label:>30}: serial {serial:>8.3} ms  parallel {parallel:>8.3} ms  {:>5.2}x  {}",
        serial / parallel,
        if expected == found {
            format!("identical {expected:016x}")
        } else {
            format!("DIFFERENT {expected:016x} {found:016x}")
        }
    );
    assert_eq!(expected, found);
}

fn grid_model(cells: usize) -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(cells, cells, cells);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    (
        (
            Grid::from((
                rubber(3.0e9, 13.0e9),
                density(),
                slab(cells, cells, 0..cells),
                &reference,
            )),
            reference,
        )
            .into(),
        coordinates,
    )
}

#[test]
#[ignore]
fn single_block() {
    println!("\nthreads: {}", threads());
    for cells in [10, 22, 46] {
        let (model, coordinates) = grid_model(cells);
        let current = deformed(&coordinates);
        let repetitions = (2_000_000 / cells.pow(3)).clamp(5, 200);
        compare(
            &format!("{} elements", cells.pow(3)),
            &model,
            &current,
            repetitions,
        );
    }
}

#[test]
#[ignore]
fn single_block_steps() {
    println!("\nthreads: {}", threads());
    for cells in [10, 22, 46] {
        let (model, coordinates) = grid_model(cells);
        let current = deformed(&coordinates);
        let masses = model.nodal_lumped_masses();
        let dt = 1e-8;
        let time = [Time::seconds(0.0), Time::seconds(STEPS as f64 * dt)];
        let velocities = || NodalVelocities::from(vec![[0.0; 3]; coordinates.len()]);
        let integrator = VelocityVerlet::new(Time::seconds(dt));
        let conditions = BoundaryConditions::<Solid<3>>::new();
        let start = Instant::now();
        let serial = model
            .integrate(
                Execution::Serial,
                &integrator,
                &time,
                (current.clone(), velocities()),
                &(),
                &masses,
                &conditions,
            )
            .unwrap();
        let serial_time = start.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
        let start = Instant::now();
        let parallel = model
            .integrate(
                Execution::Threads(threads()),
                &integrator,
                &time,
                (current.clone(), velocities()),
                &(),
                &masses,
                &conditions,
            )
            .unwrap();
        let parallel_time = start.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
        let same = checksum(serial.1.iter().last().unwrap())
            == checksum(parallel.1.iter().last().unwrap());
        println!(
            "{:>9} elements: step serial {serial_time:>8.3} ms  parallel {parallel_time:>8.3} ms  {:>5.2}x  {}",
            cells.pow(3),
            serial_time / parallel_time,
            if same { "identical" } else { "DIFFERENT" }
        );
        assert!(same);
    }
}

type Three = Blocks<Grid, Blocks<Steel, Grid>>;

const COLUMNS: usize = 32;

/// Three slabs of about 100k, 30k and 1k elements, with different constitutive models, that share
/// the nodes on their interfaces.
fn three_blocks() -> (Model<Three, 3>, Vec<[f64; 3]>) {
    let (first, second) = (98, 29);
    let layers = first + second + 1;
    let coordinates = lattice(COLUMNS, COLUMNS, layers);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let blocks = Blocks(
        Grid::from((
            rubber(3.0e9, 13.0e9),
            density(),
            slab(COLUMNS, COLUMNS, 0..first),
            &reference,
        )),
        Blocks(
            Steel::from((
                SaintVenantKirchhoff {
                    bulk_modulus: Stress::pascals(160.0e9),
                    shear_modulus: Stress::pascals(80.0e9),
                },
                density(),
                slab(COLUMNS, COLUMNS, first..first + second),
                &reference,
            )),
            Grid::from((
                rubber(1.0e9, 5.0e9),
                density(),
                slab(COLUMNS, COLUMNS, first + second..layers),
                &reference,
            )),
        ),
    );
    ((blocks, reference).into(), coordinates)
}

type Four = Blocks<Grid, Blocks<Grid, Blocks<Grid, Grid>>>;

/// Four small slabs of 1, 2, 3 and 4 layers, about 1k to 4k elements each.
fn four_blocks() -> (Model<Four, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(COLUMNS, COLUMNS, 10);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = |shear: f64, layers: Range<usize>| {
        Grid::from((
            rubber(shear, 4.0 * shear),
            density(),
            slab(COLUMNS, COLUMNS, layers),
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
fn heterogeneous_blocks() {
    println!("\nthreads: {}", threads());
    let (model, coordinates) = three_blocks();
    compare(
        "100,352 + 29,696 + 1,024",
        &model,
        &deformed(&coordinates),
        30,
    );
    let (model, coordinates) = four_blocks();
    compare(
        "1,024 + 2,048 + 3,072 + 4,096",
        &model,
        &deformed(&coordinates),
        300,
    );
}

/// The cost of each operation of a step that is not the nodal forces, on the nodes of the grid.
#[test]
#[ignore]
fn step_breakdown() {
    let cells = 46;
    let (model, coordinates) = grid_model(cells);
    let nodes = coordinates.len();
    let x = deformed(&coordinates);
    let v = NodalVelocities::from(vec![[1.0; 3]; nodes]);
    let a = NodalAccelerations::from(vec![[2.0; 3]; nodes]);
    let masses = model.nodal_lumped_masses();
    let forces = NodalForcesSolid::<3>::from(vec![[3.0; 3]; nodes]);
    let dt = Time::seconds(1e-8);
    let half = Time::seconds(0.5e-8);
    let v_half = &a * half + &v;
    println!(
        "\n{nodes} nodes ({:.1} MB per vector)",
        (3 * 8 * nodes) as f64 / 1e6
    );
    let rows = [
        (
            "v_half = a dt/2 + v",
            time_of(100, || {
                black_box(black_box(&a) * half + &v);
            }),
        ),
        (
            "x_next = v_half dt + x",
            time_of(100, || {
                black_box(black_box(&v_half) * dt + &x);
            }),
        ),
        (
            "v = a dt/2 + v_half",
            time_of(100, || {
                black_box(black_box(&a) * half + &v_half);
            }),
        ),
        (
            "clone (one history push)",
            time_of(100, || {
                black_box(black_box(&x).clone());
            }),
        ),
        (
            "zero forces",
            time_of(100, || {
                black_box(NodalForcesSolid::<3>::zero(black_box(nodes)));
            }),
        ),
        (
            "accelerations from forces",
            time_of(100, || {
                black_box(InverseMass::<3>::nodal_accelerations(
                    black_box(&masses),
                    &forces,
                ));
            }),
        ),
    ];
    rows.iter()
        .for_each(|(label, time)| println!("{label:>28}: {time:>7.3} ms"));
    let total =
        rows[0].1 + rows[1].1 + rows[2].1 + 3.0 * rows[3].1 + rows[3].1 + rows[4].1 + rows[5].1;
    println!(
        "{:>28}: {total:>7.3} ms  (3 updates + 3 history clones + closure clone + zero + accelerations)",
        "sum"
    );
}

/// The cost of a step apart from the one-time cost of a run, from runs of different lengths.
#[test]
#[ignore]
fn step_marginal() {
    println!("\nthreads: {}", threads());
    for cells in [22, 46] {
        let (model, coordinates) = grid_model(cells);
        let current = deformed(&coordinates);
        let masses = model.nodal_lumped_masses();
        let dt = 1e-8;
        let conditions = BoundaryConditions::<Solid<3>>::new();
        let integrator = VelocityVerlet::new(Time::seconds(dt));
        let run = |execution: Execution, steps: usize| {
            let time = [Time::seconds(0.0), Time::seconds(steps as f64 * dt)];
            time_of(5, || {
                black_box(
                    model
                        .integrate(
                            execution,
                            &integrator,
                            &time,
                            (
                                current.clone(),
                                NodalVelocities::from(vec![[0.0; 3]; coordinates.len()]),
                            ),
                            &(),
                            &masses,
                            &conditions,
                        )
                        .unwrap(),
                );
            })
        };
        let forces = {
            let workers = Workers::new(threads()).unwrap();
            let mut scratch = model.scratch();
            time_of(30, || {
                let mut forces = NodalForcesSolid::<3>::zero(current.len());
                model
                    .nodal_forces_parallel_into(&mut scratch, &workers, &current, &mut forces)
                    .unwrap();
                black_box(forces);
            })
        };
        for (label, execution) in [
            ("serial", Execution::Serial),
            ("threads", Execution::Threads(threads())),
        ] {
            let (short, long) = (run(execution, 20), run(execution, 120));
            let marginal = (long - short) / 100.0;
            let setup = short - 20.0 * marginal;
            println!(
                "{:>7} elements, {label:>7}: {marginal:>8.3} ms per step, one-time {setup:>8.3} ms{}",
                cells.pow(3),
                if matches!(execution, Execution::Serial) {
                    String::new()
                } else {
                    format!("   (forces alone {forces:.3} ms)")
                }
            );
        }
    }
}

/// The cost per step of the integrator alone, with a constant acceleration, from runs of
/// different lengths. The function costs one clone of the acceleration.
#[test]
#[ignore]
fn integrator_marginal() {
    use conspire::{
        fem::{NodalAccelerationsHistory, NodalCoordinatesHistory, NodalVelocitiesHistory},
        math::integrate::ExplicitDynamics,
    };
    for cells in [22, 46] {
        let (_, coordinates) = grid_model(cells);
        let nodes = coordinates.len();
        let x = deformed(&coordinates);
        let v = NodalVelocities::from(vec![[0.0; 3]; nodes]);
        let a = NodalAccelerations::from(vec![[1.0; 3]; nodes]);
        let dt = 1e-8;
        let integrator = VelocityVerlet::new(Time::seconds(dt));
        let run = |steps: usize| {
            let time = [Time::seconds(0.0), Time::seconds(steps as f64 * dt)];
            time_of(5, || {
                let result: Result<
                    (
                        _,
                        NodalCoordinatesHistory<3>,
                        NodalVelocitiesHistory<3>,
                        NodalAccelerationsHistory<3>,
                    ),
                    _,
                > = ExplicitDynamics::integrate(
                    &integrator,
                    |_, _: &NodalCoordinates<3>, _: &NodalVelocities<3>| Ok(a.clone()),
                    &time,
                    x.clone(),
                    v.clone(),
                );
                black_box(result.unwrap());
            })
        };
        let (short, long) = (run(20), run(120));
        println!(
            "{:>7} nodes: integrator alone {:>7.3} ms per step (including one clone of the acceleration)",
            nodes,
            (long - short) / 100.0
        );
    }
}
