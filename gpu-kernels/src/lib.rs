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
