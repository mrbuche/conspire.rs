use super::{GpuError, GpuGradients, GpuOptions, GpuPrecision, Resident};
use crate::{
    constitutive::solid::{elastic::GpuModel, hyperelastic::NeoHookean},
    domain::solid::{dynamics::test::fixed_dofs, mass::MassMatrix},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        solid::{
            DeviceExplicitDynamics, ElasticExplicitDynamics, Execution, Lumped, MassChoice,
            NodalForcesSolid, Unchecked,
            elastic::{ElasticElements, ParallelElasticElements},
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use std::{fs, path::Path, process::Command};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

const NODES: usize = 8;
const POINTS: usize = 8;
const REFERENCE_SIZE: usize = POINTS * NODES * 3 + POINTS;

/// Plain C++ in place of what CUDA provides, so that the kernels run on the host.
const SHIM: &str = r#"
#include <cmath>
#include <cstddef>
#define __global__
// The reciprocal of the cube root, which CUDA has and the host does not.
template <class T>
static T rcbrt(T x)
{
    return 1 / std::cbrt(x);
}
struct Index { unsigned int x; };
static Index blockIdx, blockDim, threadIdx;
static unsigned int atomicMin(unsigned int *address, unsigned int value)
{
    const unsigned int old = *address;
    if (value < old)
    {
        *address = value;
    }
    return old;
}
"#;

/// Launches the kernels over every thread of every block, one after another.
///
/// With no steps it evaluates the forces once, and with some it takes them as the device does,
/// with the same kernels in the same order.
const HARNESS: &str = r#"
#include "shim.h"
#include "kernel.cu"
#include "dynamics.cu"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

template <class T>
static std::vector<T> read(FILE *file, size_t count)
{
    std::vector<T> values(count);
    if (count && fread(values.data(), sizeof(T), count, file) != count)
    {
        std::abort();
    }
    return values;
}

template <class F>
static void launch(unsigned int threads, F kernel)
{
    blockDim.x = 128;
    for (unsigned int block = 0; block < (threads + 127) / 128; ++block)
    {
        for (unsigned int thread = 0; thread < 128; ++thread)
        {
            blockIdx.x = block;
            threadIdx.x = thread;
            kernel();
        }
    }
}

int main(int, char **argv)
{
    FILE *input = fopen(argv[1], "rb");
    const auto header = read<uint32_t>(input, 4);
    const unsigned int items = header[0], nodes = header[1], gathered = header[2], steps = header[3];
    const auto connectivity = read<uint32_t>(input, (size_t)items * NODES);
    const auto offsets = read<uint32_t>(input, gathered + 1);
    const auto contributions = read<uint32_t>(input, offsets[gathered]);
#ifdef ON_THE_FLY
    const auto derivatives = read<REAL>(input, POINTS * NODES * 3);
    const auto parametric_weights = read<REAL>(input, POINTS);
    const auto reference_coordinates = read<REAL>(input, 3 * (size_t)nodes);
#else
    const auto reference = read<REAL>(input, (size_t)items * REFERENCE_SIZE);
#endif
    auto coordinates = read<REAL>(input, 3 * (size_t)nodes);
    const auto moduli = read<REAL>(input, 2);
    const REAL dt = read<REAL>(input, 1)[0];
    const auto masses = read<REAL>(input, 3 * (size_t)nodes);
    auto velocities = read<REAL>(input, 3 * (size_t)nodes);
    fclose(input);
    const unsigned int count = 3 * nodes;
    std::vector<REAL> per_item((size_t)items * NODES * 3), forces(count, 0), accelerations(count, 0);
    unsigned int failure = 0xffffffffu;
    auto evaluate = [&]()
    {
        launch(items, [&]()
               { item_forces(items, connectivity.data(),
#ifdef ON_THE_FLY
                             reference_coordinates.data(), derivatives.data(), parametric_weights.data(),
#else
                             reference.data(),
#endif
                             coordinates.data(), per_item.data(), &failure, moduli[0], moduli[1]); });
        launch(gathered, [&]()
               { gather(gathered, items, offsets.data(), contributions.data(), per_item.data(), forces.data()); });
    };
    evaluate();
    if (steps > 0)
    {
        launch(count, [&]()
               { accelerate(count, forces.data(), masses.data(), accelerations.data(), velocities.data(), dt, 0); });
        for (unsigned int step = 0; step < steps; ++step)
        {
            launch(count, [&]()
                   { kick_drift(count, coordinates.data(), velocities.data(), accelerations.data(), dt); });
            evaluate();
            launch(count, [&]()
                   { accelerate(count, forces.data(), masses.data(), accelerations.data(), velocities.data(), dt, 1); });
        }
    }
    FILE *output = fopen(argv[2], "wb");
    fwrite(&failure, sizeof(failure), 1, output);
    if (steps == 0)
    {
        fwrite(forces.data(), sizeof(REAL), forces.size(), output);
    }
    else
    {
        fwrite(coordinates.data(), sizeof(REAL), coordinates.size(), output);
        fwrite(velocities.data(), sizeof(REAL), velocities.size(), output);
    }
    fclose(output);
    return 0;
}
"#;

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

/// A smooth displacement, large enough that every element has a stress that is not small.
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

/// The components of the nodes, as `3 * node + component`.
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

fn largest_difference(found: &[f64], expected: &[f64]) -> (f64, f64) {
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

#[test]
fn block_is_described_to_the_device() {
    let cells = 3;
    let (model, _) = grid(cells);
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    assert_eq!(model.gpu_block_count(), 1);
    assert_eq!(blocks.len(), 1);
    let block = &blocks[0];
    assert_eq!(block.number_of_items(), cells.pow(3));
    assert_eq!(block.nodes_per_item, NODES);
    assert_eq!(block.points_per_item, POINTS);
    assert_eq!(block.reference.len(), cells.pow(3) * REFERENCE_SIZE);
    assert_eq!(block.offsets.len(), (cells + 1).pow(3) + 1);
    assert_eq!(
        *block.offsets.last().unwrap() as usize,
        cells.pow(3) * NODES
    );
    assert_eq!(block.contributions.len(), cells.pow(3) * NODES);
    // The weights of the points add up to the volume of the grid, which is one.
    let volume = block
        .reference
        .chunks(REFERENCE_SIZE)
        .map(|item| item[POINTS * NODES * 3..].iter().sum::<f64>())
        .sum::<f64>();
    assert!((volume - 1.0).abs() < 1e-12);
    // The gradient vectors of the nodes at a point add up to nothing, since the shape functions
    // add up to one.
    block.reference.chunks(REFERENCE_SIZE).for_each(|item| {
        (0..POINTS).for_each(|point| {
            (0..3).for_each(|component| {
                let sum = (0..NODES)
                    .map(|node| item[(point * NODES + node) * 3 + component])
                    .sum::<f64>();
                assert!(sum.abs() < 1e-12);
            })
        })
    });
    // Every node is contributed to by the elements that have it, once each.
    block
        .contributions
        .iter()
        .for_each(|&contribution| assert!((contribution as usize) < cells.pow(3) * NODES));
}

#[test]
fn device_layout_puts_the_element_last() {
    let (model, _) = grid(3);
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    let block = &blocks[0];
    let items = block.number_of_items();
    let layout = block.device_layout();
    assert_eq!(layout.connectivity.len(), block.connectivity.len());
    assert_eq!(layout.reference.len(), block.reference.len());
    assert_eq!(layout.contributions.len(), block.contributions.len());
    for item in 0..items {
        for position in 0..NODES {
            assert_eq!(
                layout.connectivity[position * items + item],
                block.connectivity[item * NODES + position]
            );
        }
        for value in 0..REFERENCE_SIZE {
            assert_eq!(
                layout.reference[value * items + item],
                block.reference[item * REFERENCE_SIZE + value]
            );
        }
    }
    // A contribution names the same element and position as before, as position then element.
    block
        .contributions
        .iter()
        .zip(&layout.contributions)
        .for_each(|(&before, &after)| {
            let (item, position) = (before as usize / NODES, before as usize % NODES);
            assert_eq!(after as usize, position * items + item);
        });
}

/// Runs the kernels on the host as C++, from the blocks of the model, and returns whether an
/// element failed and what the harness wrote: the forces if there are no steps, and otherwise
/// the coordinates and then the velocities at the end.
///
/// Nothing is returned when there is no C++ compiler.
#[expect(clippy::too_many_arguments)]
fn run_on_host(
    name: &str,
    gradients: GpuGradients,
    model: &Model<Grid, 3>,
    coordinates: &NodalCoordinates<3>,
    velocities: &[f64],
    masses: &[f64],
    dt: f64,
    steps: usize,
) -> Option<(u32, Vec<f64>)> {
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    let block = &blocks[0];
    let directory = std::env::temp_dir().join(format!("conspire-{name}-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("shim.h"), SHIM).unwrap();
    fs::write(directory.join("kernel.cu"), include_str!("kernel.cu")).unwrap();
    fs::write(directory.join("dynamics.cu"), include_str!("dynamics.cu")).unwrap();
    fs::write(directory.join("harness.cpp"), HARNESS).unwrap();
    let on_the_fly = gradients == GpuGradients::OnTheFly;
    let compiled = Command::new("c++")
        .current_dir(&directory)
        .args([
            "-O2",
            "-ffp-contract=off",
            "-DREAL=double",
            "-DDOUBLE_PRECISION",
        ])
        .args(on_the_fly.then_some("-DON_THE_FLY"))
        .args([
            &format!("-DNODES={NODES}"),
            &format!("-DPOINTS={POINTS}"),
            "-o",
            "harness",
            "harness.cpp",
        ])
        .output();
    let Ok(compiled) = compiled else {
        println!("skipped: there is no C++ compiler");
        return None;
    };
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let mut input = Vec::new();
    let mut push = |bytes: &[u8]| input.extend_from_slice(bytes);
    push(&(block.number_of_items() as u32).to_le_bytes());
    push(&(coordinates.len() as u32).to_le_bytes());
    push(&((block.offsets.len() - 1) as u32).to_le_bytes());
    push(&(steps as u32).to_le_bytes());
    // What the kernels are given is the layout that the device has, which the harness reads.
    let layout = block.device_layout();
    layout
        .connectivity
        .iter()
        .for_each(|x| push(&x.to_le_bytes()));
    block.offsets.iter().for_each(|x| push(&x.to_le_bytes()));
    layout
        .contributions
        .iter()
        .for_each(|x| push(&x.to_le_bytes()));
    if on_the_fly {
        let shape = block.shape.as_ref().expect("the shape functions");
        shape
            .derivatives
            .iter()
            .chain(&shape.weights)
            .for_each(|x| push(&x.to_le_bytes()));
        components(&model.coordinates)
            .iter()
            .for_each(|x| push(&x.to_le_bytes()));
    } else {
        layout.reference.iter().for_each(|x| push(&x.to_le_bytes()));
    }
    components(coordinates)
        .iter()
        .for_each(|x| push(&x.to_le_bytes()));
    let GpuModel::NeoHookean {
        shear_modulus,
        bulk_modulus,
    } = block.model;
    push(&shear_modulus.to_le_bytes());
    push(&bulk_modulus.to_le_bytes());
    push(&dt.to_le_bytes());
    masses.iter().for_each(|x| push(&x.to_le_bytes()));
    velocities.iter().for_each(|x| push(&x.to_le_bytes()));
    fs::write(directory.join("input.bin"), input).unwrap();
    let ran = Command::new(directory.join("harness"))
        .args([Path::new("input.bin"), Path::new("output.bin")])
        .current_dir(&directory)
        .status()
        .unwrap();
    assert!(ran.success());
    let output = fs::read(directory.join("output.bin")).unwrap();
    let failure = u32::from_le_bytes(output[..4].try_into().unwrap());
    let found = output[4..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&bytes| f64::from_le_bytes(bytes))
        .collect::<Vec<_>>();
    let _ = fs::remove_dir_all(&directory);
    Some((failure, found))
}

/// The kernels, run on the host as C++, give the forces that the library gives.
///
/// Skipped when there is no C++ compiler. This checks the arithmetic of the kernels and the
/// layout of what they are given, and needs no device.
#[test]
fn kernels_on_the_host_give_the_forces_of_the_library() {
    let (model, coordinates) = grid(4);
    let current = deformed(&coordinates);
    let zeros = vec![0.0; 3 * current.len()];
    let ones = vec![1.0; 3 * current.len()];
    let expected = components(&model.nodal_forces(&current).unwrap());
    for (name, gradients) in [
        ("forces-stored", GpuGradients::Stored),
        ("forces-fly", GpuGradients::OnTheFly),
    ] {
        let Some((failure, found)) =
            run_on_host(name, gradients, &model, &current, &zeros, &ones, 0.0, 0)
        else {
            return;
        };
        assert_eq!(failure, u32::MAX, "an element was found to be inverted");
        let (difference, scale) = largest_difference(&found, &expected);
        assert!(scale > 1e3, "the forces are not small enough to be nothing");
        assert!(
            difference < 1e-10 * scale,
            "{gradients:?}: the largest difference is {difference} of forces up to {scale}"
        );
    }
}

/// The degrees of freedom held in the runs below: the bottom layer of nodes of the grid in
/// every direction, and the top layer in one, so that the body is neither free nor still.
fn held(cells: usize) -> Vec<usize> {
    let layer = (cells + 1) * (cells + 1);
    let nodes = (cells + 1) * layer;
    (0..layer)
        .flat_map(|node| [3 * node, 3 * node + 1, 3 * node + 2])
        .chain((nodes - layer..nodes).map(|node| 3 * node + 2))
        .collect()
}

/// A velocity that varies from node to node.
fn initial_velocities(coordinates: &[[f64; 3]]) -> NodalVelocities<3> {
    NodalVelocities::from(
        coordinates
            .iter()
            .map(|&[x, y, z]| [1.0 * y, -0.5 * z, 0.25 * x])
            .collect::<Vec<_>>(),
    )
}

const STEPS: usize = 25;
const STEP: f64 = 1e-8;

/// The state at the end of the run of the library on the calling thread.
fn serial_run(
    model: &Model<Grid, 3>,
    current: &NodalCoordinates<3>,
    velocities: &NodalVelocities<3>,
    held: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let run = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (current.clone(), velocities.clone()),
            &(),
            &fixed_dofs(held),
        )
        .unwrap();
    (
        components(run.1.iter().last().unwrap()),
        components(run.2.iter().last().unwrap()),
    )
}

/// The steps of the kernels, run on the host as C++, are the steps of the library, with some
/// degrees of freedom held.
///
/// Skipped when there is no C++ compiler. This checks the loop that stays on the device, with
/// the masses and what is held, against the integrator of the library.
#[test]
fn steps_on_the_host_are_the_steps_of_the_library() {
    let cells = 4;
    let (model, coordinates) = grid(cells);
    let current = deformed(&coordinates);
    let velocities = initial_velocities(&coordinates);
    let held = held(cells);
    let masses = MassMatrix::<3>::inverse(&Lumped.masses(&model), &held)
        .unwrap()
        .per_degree_of_freedom::<3>();
    let (expected_coordinates, expected_velocities) =
        serial_run(&model, &current, &velocities, &held);
    // The library gives no velocity to what is held before a run, and the kernels are given
    // the same.
    let mut held_velocities = components(&velocities);
    held.iter().for_each(|&dof| held_velocities[dof] = 0.0);
    for (name, gradients) in [
        ("steps-stored", GpuGradients::Stored),
        ("steps-fly", GpuGradients::OnTheFly),
    ] {
        let Some((failure, found)) = run_on_host(
            name,
            gradients,
            &model,
            &current,
            &held_velocities,
            &masses,
            STEP,
            STEPS,
        ) else {
            return;
        };
        assert_eq!(failure, u32::MAX, "an element was found to be inverted");
        let count = 3 * current.len();
        let (difference, scale) = largest_difference(&found[..count], &expected_coordinates);
        assert!(
            difference < 1e-12 * scale.max(1.0),
            "{gradients:?}: the coordinates differ by {difference}"
        );
        let (difference, scale) = largest_difference(&found[count..], &expected_velocities);
        assert!(
            scale > 1e-3 && difference < 1e-9 * scale,
            "{gradients:?}: the velocities differ by {difference} of up to {scale}"
        );
        // What was held did not move.
        held.iter().for_each(|&dof| {
            assert_eq!(found[dof], current[dof / 3][dof % 3].value());
        });
    }
}

/// The device, if there is one: nothing to test without it.
fn resident(
    model: &Model<Grid, 3>,
    precision: GpuPrecision,
    gradients: GpuGradients,
) -> Option<Resident> {
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    match Resident::new(
        GpuOptions {
            precision,
            gradients,
            ..GpuOptions::default()
        },
        &blocks,
        &model.coordinates,
    ) {
        Ok(resident) => {
            println!("device: {}", resident.device_name());
            Some(resident)
        }
        Err(GpuError::Unavailable(message)) => {
            println!("skipped: no device ({message})");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn device_forces_are_those_of_the_library() {
    let (model, coordinates) = grid(6);
    let current = deformed(&coordinates);
    let expected = components(&model.nodal_forces(&current).unwrap());
    for (precision, gradients, tolerance) in [
        (GpuPrecision::F64, GpuGradients::Stored, 1e-10),
        (GpuPrecision::F64, GpuGradients::OnTheFly, 1e-10),
        (GpuPrecision::F32, GpuGradients::Stored, 1e-4),
        (GpuPrecision::F32, GpuGradients::OnTheFly, 1e-4),
    ] {
        let Some(mut resident) = resident(&model, precision, gradients) else {
            return;
        };
        let mut forces = NodalForcesSolid::<3>::zero(current.len());
        resident.nodal_forces_into(&current, &mut forces).unwrap();
        // The forces are added, so a second evaluation is twice the first.
        resident.nodal_forces_into(&current, &mut forces).unwrap();
        let found = components(&forces)
            .into_iter()
            .map(|force| 0.5 * force)
            .collect::<Vec<_>>();
        let (difference, scale) = largest_difference(&found, &expected);
        assert!(
            difference < tolerance * scale,
            "{precision:?} {gradients:?}: the largest difference is {difference} of forces up to {scale}"
        );
    }
}

#[test]
fn device_reports_the_first_inverted_element() {
    let (model, coordinates) = grid(4);
    let mut current = deformed(&coordinates);
    // Collapsing the nodes of the last element onto one point inverts it.
    let inverted = hexahedra(4).last().copied().unwrap();
    let point = current[inverted[0]].clone();
    inverted
        .iter()
        .for_each(|&node| current[node] = point.clone());
    let Some(mut resident) = resident(&model, GpuPrecision::F64, GpuGradients::Stored) else {
        return;
    };
    let mut forces = NodalForcesSolid::<3>::zero(current.len());
    match resident.nodal_forces_into(&current, &mut forces) {
        Err(GpuError::Item { block: 0, item }) => {
            assert!(model.gpu_item_error(0, item, &current).is_some())
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn run_on_the_device_follows_the_run_on_the_calling_thread() {
    let (model, coordinates) = grid(5);
    let current = deformed(&coordinates);
    let Some(_) = resident(&model, GpuPrecision::F64, GpuGradients::Stored) else {
        return;
    };
    let dt = 1e-8;
    let time = [Time::seconds(0.0), Time::seconds(20.0 * dt)];
    let integrator = VelocityVerlet::new(Time::seconds(dt));
    let conditions = fixed_dofs(&[]);
    let run = |execution| {
        model
            .integrate(
                execution,
                Unchecked,
                Lumped,
                &integrator,
                &time,
                (
                    current.clone(),
                    NodalVelocities::from(vec![[0.0; 3]; current.len()]),
                ),
                &(),
                &conditions,
            )
            .unwrap()
    };
    let serial = run(Execution::Serial);
    let last = |history: &crate::fem::NodalCoordinatesHistory<3>| {
        components(history.iter().last().unwrap())
    };
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let device = run(Execution::Gpu(GpuOptions {
            gradients,
            ..GpuOptions::default()
        }));
        let (difference, scale) = largest_difference(&last(&device.1), &last(&serial.1));
        assert!(
            difference < 1e-9 * scale.max(1.0),
            "{gradients:?}: {difference}"
        );
    }
}

#[test]
fn run_that_stays_on_the_device_follows_the_run_on_the_calling_thread() {
    let cells = 4;
    let (model, coordinates) = grid(cells);
    let current = deformed(&coordinates);
    let velocities = initial_velocities(&coordinates);
    let held = held(cells);
    let Some(_) = resident(&model, GpuPrecision::F64, GpuGradients::Stored) else {
        return;
    };
    let (expected_coordinates, expected_velocities) =
        serial_run(&model, &current, &velocities, &held);
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let (found_coordinates, found_velocities) = model
            .integrate_on_device(
                GpuOptions {
                    gradients,
                    ..GpuOptions::default()
                },
                Time::seconds(STEP),
                STEPS,
                (current.clone(), velocities.clone()),
                &fixed_dofs(&held),
            )
            .unwrap();
        let (difference, scale) =
            largest_difference(&components(&found_coordinates), &expected_coordinates);
        assert!(
            difference < 1e-12 * scale.max(1.0),
            "{gradients:?}: {difference}"
        );
        let (difference, scale) =
            largest_difference(&components(&found_velocities), &expected_velocities);
        assert!(
            scale > 1e-3 && difference < 1e-9 * scale,
            "{gradients:?}: {difference}"
        );
    }
}
