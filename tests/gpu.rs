#![cfg(feature = "gpu")]

//! Explicit dynamics on a CUDA device, against the same runs on the calling thread.
//!
//! These need a device, and pass without doing anything when there is none. They use only the
//! public interface, so that running them compiles the library and this file and not the unit
//! tests of the library: `cargo test -F gpu --test gpu -- --nocapture`.

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            DeviceExplicitDynamics, ElasticExplicitDynamics, Execution, GpuGradients, GpuKernels,
            GpuOptions, GpuPrecision, Lumped, Unchecked,
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

const CELLS: usize = 6;
const STEPS: usize = 40;
const STEP: f64 = 1e-8;

fn index(i: usize, j: usize, k: usize) -> usize {
    (k * (CELLS + 1) + j) * (CELLS + 1) + i
}

fn lattice() -> Vec<[f64; 3]> {
    (0..=CELLS)
        .flat_map(|k| {
            (0..=CELLS).flat_map(move |j| {
                (0..=CELLS).map(move |i| {
                    [
                        i as f64 / CELLS as f64,
                        j as f64 / CELLS as f64,
                        k as f64 / CELLS as f64,
                    ]
                })
            })
        })
        .collect()
}

fn hexahedra() -> Vec<[usize; 8]> {
    (0..CELLS)
        .flat_map(|k| {
            (0..CELLS).flat_map(move |j| {
                (0..CELLS).map(move |i| {
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

fn model() -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    (
        (
            Grid::from((
                NeoHookean {
                    shear_modulus: Stress::pascals(3.0e9),
                    bulk_modulus: Stress::pascals(13.0e9),
                },
                Density::kilograms_per_cubic_meter(7.8e3),
                hexahedra(),
                &reference,
            )),
            reference,
        )
            .into(),
        coordinates,
    )
}

/// A smooth displacement, so that every element has a stress that is not small.
fn deformed(coordinates: &[[f64; 3]]) -> NodalCoordinates<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                x + 2e-2 * (3.0 * y).sin() + 1e-2 * z,
                y + 2e-2 * (2.0 * z).sin(),
                z + 2e-2 * (4.0 * x).sin() - 1e-2 * x,
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn velocities(coordinates: &[[f64; 3]]) -> NodalVelocities<3> {
    NodalVelocities::from(
        coordinates
            .iter()
            .map(|&[x, y, z]| [1.0 * y, -0.5 * z, 0.25 * x])
            .collect::<Vec<_>>(),
    )
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

/// The largest difference, and the largest of what it is a difference from.
fn difference(found: &[f64], expected: &[f64]) -> (f64, f64) {
    (
        found
            .iter()
            .zip(expected)
            .map(|(found, expected)| (found - expected).abs())
            .fold(0.0, f64::max),
        expected
            .iter()
            .map(|expected| expected.abs())
            .fold(0.0, f64::max),
    )
}

/// Whether an error is that there is no device, which these tests skip.
fn no_device(error: &impl std::fmt::Debug) -> bool {
    format!("{error:?}").contains("No CUDA device is available")
}

/// The run on the calling thread: the coordinates and the velocities at the end.
fn serial(
    model: &Model<Grid, 3>,
    coordinates: &NodalCoordinates<3>,
    velocities: &NodalVelocities<3>,
) -> (Vec<f64>, Vec<f64>) {
    let run = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .unwrap();
    (
        components(run.1.iter().last().unwrap()),
        components(run.2.iter().last().unwrap()),
    )
}

/// The run that copies the state at every step, as the integrator of the library does.
#[test]
fn run_with_history_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    let run = model.integrate(
        Execution::Gpu(GpuOptions::default()),
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(STEP)),
        &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
        (coordinates.clone(), velocities.clone()),
        &(),
        &BoundaryConditions::<Solid<3>>::new(),
    );
    let run = match run {
        Ok(run) => run,
        Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
        Err(error) => panic!("{error:?}"),
    };
    let (gap, scale) = difference(&components(run.1.iter().last().unwrap()), &expected.0);
    assert!(
        gap < 1e-12 * scale.max(1.0),
        "the coordinates differ by {gap}"
    );
    let (gap, scale) = difference(&components(run.2.iter().last().unwrap()), &expected.1);
    assert!(
        gap < 1e-9 * scale.max(1.0),
        "the velocities differ by {gap}"
    );
}

/// The run that stays on the device, in both precisions.
#[test]
fn run_on_the_device_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    for (precision, tolerance) in [(GpuPrecision::F64, 1e-9), (GpuPrecision::F32, 1e-3)] {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?}: the velocities differ by {gap}"
        );
    }
}

/// The run that stays on the device, with the code of the library in place of the kernels written
/// for it: the forces of the elements, the sums on the nodes and the steps, in double precision.
#[test]
fn run_with_the_code_of_the_library_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let run = model.integrate_on_device(
            GpuOptions {
                kernels: GpuKernels::Library,
                gradients,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the velocities differ by {gap}"
        );
    }
}

/// Two blocks of different materials that share the plane of nodes between them: a lattice twice
/// as long in one direction, with the cells on one side of the plane in the first block and the
/// others in the second.
fn two_blocks() -> (Model<Blocks<Grid, Grid>, 3>, Vec<[f64; 3]>) {
    let nodes_x = 2 * CELLS + 1;
    let index = move |i: usize, j: usize, k: usize| (k * (CELLS + 1) + j) * nodes_x + i;
    let coordinates = (0..=CELLS)
        .flat_map(|k| {
            (0..=CELLS).flat_map(move |j| {
                (0..nodes_x).map(move |i| {
                    [
                        i as f64 / CELLS as f64,
                        j as f64 / CELLS as f64,
                        k as f64 / CELLS as f64,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    let cells = |start: usize| {
        (0..CELLS)
            .flat_map(|k| {
                (0..CELLS).flat_map(move |j| {
                    (start..start + CELLS).map(move |i| {
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
            .collect::<Vec<_>>()
    };
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let first = Grid::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0e9),
            bulk_modulus: Stress::pascals(13.0e9),
        },
        Density::kilograms_per_cubic_meter(7.8e3),
        cells(0),
        &reference,
    ));
    let second = Grid::from((
        NeoHookean {
            shear_modulus: Stress::pascals(1.0e9),
            bulk_modulus: Stress::pascals(5.0e9),
        },
        Density::kilograms_per_cubic_meter(2.7e3),
        cells(CELLS),
        &reference,
    ));
    ((Blocks(first, second), reference).into(), coordinates)
}

/// A model of two blocks, which have different materials and share nodes, on the device in every
/// way that there is, against the run on the calling thread.
#[test]
fn run_of_a_model_of_two_blocks_follows_the_run_on_the_calling_thread() {
    let (model, reference) = two_blocks();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .unwrap();
    let expected = (
        components(expected.1.iter().last().unwrap()),
        components(expected.2.iter().last().unwrap()),
    );
    for (precision, gradients, kernels, tolerance) in [
        (
            GpuPrecision::F64,
            GpuGradients::Stored,
            GpuKernels::Handwritten,
            1e-9,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::OnTheFly,
            GpuKernels::Handwritten,
            1e-9,
        ),
        (
            GpuPrecision::F32,
            GpuGradients::Stored,
            GpuKernels::Handwritten,
            1e-3,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::Stored,
            GpuKernels::Library,
            1e-9,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::OnTheFly,
            GpuKernels::Library,
            1e-9,
        ),
    ] {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                gradients,
                kernels,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{precision:?} {gradients:?} {kernels:?}: {error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the velocities differ by {gap}"
        );
    }
}
