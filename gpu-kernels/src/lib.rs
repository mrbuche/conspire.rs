//! conspire's own code, compiled for a CUDA device.
//!
//! This is built with the nightly toolchain for `nvptx64-nvidia-cuda`, and gives PTX that the
//! driver compiles for whatever device it runs on:
//! `cargo build --release` in this directory.

#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]

use conspire::{
    constitutive::solid::{elastic::Elastic, hyperelastic::NeoHookean},
    fem::block::element::{
        ElementNodalCoordinates, linear::Hexahedron, solid::elastic::ElasticElement,
    },
    math::Tensor,
    mechanics::DeformationGradient,
    units::Stress,
};
use core::{
    alloc::{GlobalAlloc, Layout},
    arch::nvptx,
    panic::PanicInfo,
    sync::atomic::{AtomicU32, Ordering},
};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { nvptx::trap() }
}

/// A device has no heap that the code here uses, so asking for memory stops the thread.
struct NoHeap;

// SAFETY: it never gives any memory.
unsafe impl GlobalAlloc for NoHeap {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        // SAFETY: stops the thread.
        unsafe { nvptx::trap() }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static HEAP: NoHeap = NoHeap;

/// The index of the thread among all of the threads of the launch.
fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe {
        (nvptx::_block_idx_x() as usize) * (nvptx::_block_dim_x() as usize)
            + nvptx::_thread_idx_x() as usize
    }
}

/// Fills `output` with the squares of the numbers from one, to check that code made here runs.
///
/// # Safety
///
/// `output` must have `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn squares(output: *mut f64, count: u32) {
    let index = thread();
    if index < count as usize {
        let value = (index + 1) as f64;
        // SAFETY: the index is within the values that `output` has.
        unsafe { *output.add(index) = value * value };
    }
}

/// The first Piola-Kirchhoff stress of a Neo-Hookean solid, by the code of the library, for
/// each of `count` deformation gradients of nine values, row by row.
///
/// A deformation gradient that has no positive Jacobian gives a stress of zeros and a one in
/// `failures`.
///
/// # Safety
///
/// `deformation_gradients` and `stresses` must have `9 * count` values, and `failures` `count`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn neo_hookean_stress(
    deformation_gradients: *const f64,
    stresses: *mut f64,
    failures: *mut u32,
    count: u32,
    shear_modulus: f64,
    bulk_modulus: f64,
) {
    let index = thread();
    if index < count as usize {
        // SAFETY: the index is within the values that these have.
        let (input, output, failure) = unsafe {
            (
                core::slice::from_raw_parts(deformation_gradients.add(9 * index), 9),
                core::slice::from_raw_parts_mut(stresses.add(9 * index), 9),
                &mut *failures.add(index),
            )
        };
        let model = NeoHookean {
            bulk_modulus: Stress::pascals(bulk_modulus),
            shear_modulus: Stress::pascals(shear_modulus),
        };
        let deformation_gradient = DeformationGradient::from([
            [input[0], input[1], input[2]],
            [input[3], input[4], input[5]],
            [input[6], input[7], input[8]],
        ]);
        match model.first_piola_kirchhoff_stress(&deformation_gradient) {
            Ok(stress) => {
                stress.iter().enumerate().for_each(|(i, row)| {
                    row.iter()
                        .enumerate()
                        .for_each(|(j, component)| output[3 * i + j] = component.value())
                });
                *failure = 0;
            }
            Err(_) => {
                output.iter_mut().for_each(|value| *value = 0.0);
                *failure = 1;
            }
        }
    }
}

/// The nodal forces of hexahedra of a Neo-Hookean solid, by the code of the library, one element
/// to a thread.
///
/// The elements are those of the library, as the bytes that they are on the host, so each has
/// the gradient vectors and weights that it was made with. The nodes of an element are eight
/// numbers, and its forces are twenty-four values, by node and then component, for each. An
/// element whose Jacobian is not positive gives forces of zeros and a one in `failures`.
///
/// # Safety
///
/// `elements` must be `count` hexahedra, `connectivity` must be `8 * count` nodes that
/// `coordinates` has, `forces` must have `24 * count` values, and `failures` must have `count`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn neo_hookean_hexahedron_forces(
    elements: *const Hexahedron,
    connectivity: *const u32,
    coordinates: *const f64,
    forces: *mut f64,
    failures: *mut u32,
    count: u32,
    shear_modulus: f64,
    bulk_modulus: f64,
) {
    let index = thread();
    if index < count as usize {
        // SAFETY: the index is within the elements, nodes, forces and failures that there are.
        let (element, nodes, output, failure) = unsafe {
            (
                &*elements.add(index),
                core::slice::from_raw_parts(connectivity.add(8 * index), 8),
                core::slice::from_raw_parts_mut(forces.add(24 * index), 24),
                &mut *failures.add(index),
            )
        };
        let model = NeoHookean {
            bulk_modulus: Stress::pascals(bulk_modulus),
            shear_modulus: Stress::pascals(shear_modulus),
        };
        let mut positions = [[0.0; 3]; 8];
        positions
            .iter_mut()
            .zip(nodes)
            .for_each(|(position, &node)| {
                // SAFETY: the node is one that `coordinates` has.
                let node =
                    unsafe { core::slice::from_raw_parts(coordinates.add(3 * node as usize), 3) };
                position.copy_from_slice(node)
            });
        match element.nodal_forces(&model, &ElementNodalCoordinates::<8>::from(positions)) {
            Ok(nodal_forces) => {
                nodal_forces.iter().enumerate().for_each(|(a, force)| {
                    force
                        .iter()
                        .enumerate()
                        .for_each(|(i, component)| output[3 * a + i] = component.value())
                });
                *failure = 0;
            }
            Err(_) => {
                output.iter_mut().for_each(|value| *value = 0.0);
                *failure = 1;
            }
        }
    }
}

// What follows is a block of hexahedra of a Neo-Hookean solid as the run of dynamics on a device
// takes it: the forces of the elements, which are the code of the library, then the forces on the
// nodes, and the steps of velocity Verlet. The data of the elements and the forces are laid out
// with the element last, so that consecutive threads, which are consecutive elements, read
// and write consecutive addresses:
//   connectivity[position * items + item]
//   forces[(component * 8 + position) * items + item]

/// The words of a hexahedron, which are its numbers one after another as they are on the host.
const WORDS: usize = core::mem::size_of::<Hexahedron>() / 8;
const _: () = assert!(
    core::mem::size_of::<Hexahedron>() % 8 == 0 && core::mem::align_of::<Hexahedron>() == 8
);

/// The nodal forces of each of `items` hexahedra of a block, by the code of the library, one to a
/// thread. The first element whose Jacobian is not positive is put in `failure`, which is
/// all ones when none is, and its forces are zeros.
///
/// The elements are the words of each, with the element last, `elements[word * items + item]`,
/// so that consecutive threads read consecutive addresses, and a thread puts its words back into
/// the hexahedron that the library takes. The host says how many words an element has, and the
/// thread stops if that is not what this was built for.
///
/// # Safety
///
/// `elements` must be `words * items` words of hexahedra, `connectivity` must be `8 * items`
/// nodes that `coordinates` has, `forces` must have `24 * items` values, and `failure` must be
/// one.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn block_forces(
    items: u32,
    words: u32,
    elements: *const u64,
    connectivity: *const u32,
    coordinates: *const f64,
    forces: *mut f64,
    failure: *mut u32,
    shear_modulus: f64,
    bulk_modulus: f64,
) {
    let item = thread();
    let items = items as usize;
    if item < items {
        if words as usize != WORDS {
            // SAFETY: stops the thread.
            unsafe { nvptx::trap() }
        }
        let mut stored = core::mem::MaybeUninit::<Hexahedron>::uninit();
        let target = stored.as_mut_ptr().cast::<u64>();
        (0..WORDS).for_each(|word| {
            // SAFETY: the word is one of the words of the element, which is as many as the
            // hexahedron has, and it is a plain number.
            unsafe { *target.add(word) = *elements.add(word * items + item) }
        });
        // SAFETY: every word of the hexahedron was put there, and its numbers are plain.
        let element = unsafe { stored.assume_init_ref() };
        let mut positions = [[0.0; 3]; 8];
        positions.iter_mut().enumerate().for_each(|(a, position)| {
            // SAFETY: the node is one that `coordinates` has.
            let node = unsafe { *connectivity.add(a * items + item) } as usize;
            (0..3).for_each(|i| position[i] = unsafe { *coordinates.add(3 * node + i) });
        });
        let model = NeoHookean {
            bulk_modulus: Stress::pascals(bulk_modulus),
            shear_modulus: Stress::pascals(shear_modulus),
        };
        let mut nodal_forces = [[0.0; 3]; 8];
        match element.nodal_forces(&model, &ElementNodalCoordinates::<8>::from(positions)) {
            Ok(computed) => computed.iter().enumerate().for_each(|(a, force)| {
                force
                    .iter()
                    .enumerate()
                    .for_each(|(i, component)| nodal_forces[a][i] = component.value())
            }),
            Err(_) => {
                // SAFETY: `failure` is one value, which the threads only take the least of.
                unsafe { &*(failure as *const AtomicU32) }
                    .fetch_min(item as u32, Ordering::Relaxed);
            }
        }
        nodal_forces.iter().enumerate().for_each(|(a, force)| {
            force.iter().enumerate().for_each(|(i, &component)| {
                // SAFETY: the index is within the forces that there are.
                unsafe { *forces.add((i * 8 + a) * items + item) = component }
            })
        });
    }
}

/// The forces on `nodes` nodes, each the sum of the forces of the elements that have it, in the
/// order of the elements, added to the forces that it has.
///
/// A contribution is `position * items + item`, which is where the force of an element on a
/// node is in each plane of components.
///
/// # Safety
///
/// The offsets must be `nodes + 1` and the contributions what they say, within the forces of
/// the elements, which are `24 * items`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gather(
    nodes: u32,
    items: u32,
    offsets: *const u32,
    contributions: *const u32,
    forces: *const f64,
    nodal_forces: *mut f64,
) {
    let node = thread();
    if node < nodes as usize {
        let plane = 8 * items as usize;
        // SAFETY: the node is within the nodes that there are.
        let (first, last, mut sum) = unsafe {
            (
                *offsets.add(node),
                *offsets.add(node + 1),
                [
                    *nodal_forces.add(3 * node),
                    *nodal_forces.add(3 * node + 1),
                    *nodal_forces.add(3 * node + 2),
                ],
            )
        };
        for k in first..last {
            // SAFETY: the contribution is within the forces of the elements.
            let contribution = unsafe { *contributions.add(k as usize) } as usize;
            sum.iter_mut()
                .enumerate()
                .for_each(|(i, value)| *value += unsafe { *forces.add(i * plane + contribution) });
        }
        sum.iter()
            .enumerate()
            .for_each(|(i, &value)| unsafe { *nodal_forces.add(3 * node + i) = value });
    }
}

/// The first half of a step of velocity Verlet: the velocity a half step on, and the coordinates
/// a step on with it.
///
/// # Safety
///
/// Each of the arrays must have `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kick_drift(
    count: u32,
    coordinates: *mut f64,
    velocities: *mut f64,
    accelerations: *const f64,
    dt: f64,
) {
    let index = thread();
    if index < count as usize {
        // SAFETY: the index is within the values that there are.
        unsafe {
            let half_velocity = *accelerations.add(index) * (dt * 0.5) + *velocities.add(index);
            *velocities.add(index) = half_velocity;
            *coordinates.add(index) = half_velocity * dt + *coordinates.add(index);
        }
    }
}

/// The accelerations that balance the forces, which are then cleared for the next step, and the
/// velocity another half step on when `kick` is not zero.
///
/// # Safety
///
/// Each of the arrays must have `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn accelerate(
    count: u32,
    forces: *mut f64,
    masses: *const f64,
    accelerations: *mut f64,
    velocities: *mut f64,
    dt: f64,
    kick: u32,
) {
    let index = thread();
    if index < count as usize {
        // SAFETY: the index is within the values that there are.
        unsafe {
            let acceleration = -*forces.add(index) / *masses.add(index);
            *forces.add(index) = 0.0;
            *accelerations.add(index) = acceleration;
            if kick != 0 {
                *velocities.add(index) = acceleration * (dt * 0.5) + *velocities.add(index);
            }
        }
    }
}
