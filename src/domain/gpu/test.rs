use super::{GpuError, GpuOptions, GpuPrecision, Resident};
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{Block, element::linear::Hexahedron},
        boundary::{BoundaryConditions, Solid},
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, NodalForcesSolid, Unchecked,
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
const HARNESS: &str = r#"
#include "shim.h"
#include "kernel.cu"
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

int main(int, char **argv)
{
    FILE *input = fopen(argv[1], "rb");
    const auto header = read<uint32_t>(input, 3);
    const unsigned int items = header[0], nodes = header[1], gathered = header[2];
    const auto connectivity = read<uint32_t>(input, (size_t)items * NODES);
    const auto offsets = read<uint32_t>(input, gathered + 1);
    const auto contributions = read<uint32_t>(input, offsets[gathered]);
    const auto reference = read<REAL>(input, (size_t)items * REFERENCE_SIZE);
    const auto coordinates = read<REAL>(input, 3 * (size_t)nodes);
    const auto moduli = read<REAL>(input, 2);
    fclose(input);
    std::vector<REAL> per_item((size_t)items * NODES * 3), forces(3 * (size_t)nodes, 0);
    unsigned int failure = 0xffffffffu;
    blockDim.x = 128;
    for (unsigned int block = 0; block < (items + 127) / 128; ++block)
    {
        for (unsigned int thread = 0; thread < 128; ++thread)
        {
            blockIdx.x = block;
            threadIdx.x = thread;
            item_forces(items, connectivity.data(), reference.data(), coordinates.data(),
                               per_item.data(), &failure, moduli[0], moduli[1]);
        }
    }
    for (unsigned int block = 0; block < (gathered + 127) / 128; ++block)
    {
        for (unsigned int thread = 0; thread < 128; ++thread)
        {
            blockIdx.x = block;
            threadIdx.x = thread;
            gather(gathered, offsets.data(), contributions.data(), per_item.data(), forces.data());
        }
    }
    FILE *output = fopen(argv[2], "wb");
    fwrite(&failure, sizeof(failure), 1, output);
    fwrite(forces.data(), sizeof(REAL), forces.size(), output);
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

fn components(forces: &NodalForcesSolid<3>) -> Vec<f64> {
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value()))
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

/// The kernels, run on the host as C++, give the forces that the library gives.
///
/// Skipped when there is no C++ compiler. This checks the arithmetic of the kernels and the
/// layout of what they are given, and needs no device.
#[test]
fn kernels_on_the_host_give_the_forces_of_the_library() {
    let cells = 4;
    let (model, coordinates) = grid(cells);
    let current = deformed(&coordinates);
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    let block = &blocks[0];
    let directory = std::env::temp_dir().join(format!("conspire-kernels-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("shim.h"), SHIM).unwrap();
    fs::write(directory.join("kernel.cu"), include_str!("kernel.cu")).unwrap();
    fs::write(directory.join("harness.cpp"), HARNESS).unwrap();
    let compiled = Command::new("c++")
        .current_dir(&directory)
        .args([
            "-O2",
            "-ffp-contract=off",
            "-DREAL=double",
            &format!("-DNODES={NODES}"),
            &format!("-DPOINTS={POINTS}"),
            "-o",
            "harness",
            "harness.cpp",
        ])
        .output();
    let Ok(compiled) = compiled else {
        println!("skipped: there is no C++ compiler");
        return;
    };
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let mut input = Vec::new();
    let mut push = |bytes: &[u8]| input.extend_from_slice(bytes);
    push(&(block.number_of_items() as u32).to_le_bytes());
    push(&(current.len() as u32).to_le_bytes());
    push(&((block.offsets.len() - 1) as u32).to_le_bytes());
    block
        .connectivity
        .iter()
        .for_each(|x| push(&x.to_le_bytes()));
    block.offsets.iter().for_each(|x| push(&x.to_le_bytes()));
    block
        .contributions
        .iter()
        .for_each(|x| push(&x.to_le_bytes()));
    block.reference.iter().for_each(|x| push(&x.to_le_bytes()));
    current
        .iter()
        .for_each(|c| c.iter().for_each(|x| push(&x.value().to_le_bytes())));
    let crate::constitutive::solid::elastic::GpuModel::NeoHookean {
        shear_modulus,
        bulk_modulus,
    } = block.model;
    push(&shear_modulus.to_le_bytes());
    push(&bulk_modulus.to_le_bytes());
    fs::write(directory.join("input.bin"), input).unwrap();
    let ran = Command::new(directory.join("harness"))
        .args([Path::new("input.bin"), Path::new("output.bin")])
        .current_dir(&directory)
        .status()
        .unwrap();
    assert!(ran.success());
    let output = fs::read(directory.join("output.bin")).unwrap();
    let failure = u32::from_le_bytes(output[..4].try_into().unwrap());
    assert_eq!(failure, u32::MAX, "an element was found to be inverted");
    let found = output[4..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&bytes| f64::from_le_bytes(bytes))
        .collect::<Vec<_>>();
    let expected = components(&model.nodal_forces(&current).unwrap());
    let (difference, scale) = largest_difference(&found, &expected);
    assert!(scale > 1e3, "the forces are not small enough to be nothing");
    assert!(
        difference < 1e-10 * scale,
        "the largest difference is {difference} of forces up to {scale}"
    );
    let _ = fs::remove_dir_all(&directory);
}

/// The device, if there is one: nothing to test without it.
fn resident(model: &Model<Grid, 3>, precision: GpuPrecision) -> Option<Resident> {
    let mut blocks = Vec::new();
    model.gpu_blocks(&mut blocks).unwrap();
    match Resident::new(
        GpuOptions {
            precision,
            ..GpuOptions::default()
        },
        &blocks,
        model.coordinates.len(),
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
    for (precision, tolerance) in [(GpuPrecision::F64, 1e-10), (GpuPrecision::F32, 1e-4)] {
        let Some(mut resident) = resident(&model, precision) else {
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
            "{precision:?}: the largest difference is {difference} of forces up to {scale}"
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
    let Some(mut resident) = resident(&model, GpuPrecision::F64) else {
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
    let Some(_) = resident(&model, GpuPrecision::F64) else {
        return;
    };
    let dt = 1e-8;
    let time = [Time::seconds(0.0), Time::seconds(20.0 * dt)];
    let integrator = VelocityVerlet::new(Time::seconds(dt));
    let conditions = BoundaryConditions::<Solid<3>>::new();
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
    let device = run(Execution::Gpu(GpuOptions::default()));
    let last = |history: &crate::fem::NodalCoordinatesHistory<3>| {
        history
            .iter()
            .last()
            .unwrap()
            .iter()
            .flat_map(|coordinate| coordinate.iter().map(|component| component.value()))
            .collect::<Vec<_>>()
    };
    let (difference, scale) = largest_difference(&last(&device.1), &last(&serial.1));
    assert!(difference < 1e-9 * scale.max(1.0), "{difference}");
}
