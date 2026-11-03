use super::super::{
    Gradient, SERIAL,
    element::bounded,
    kind::KINDS,
    test::{distorted, grid, offset, terms, typed},
};
use super::{Energy, Reach, displacements, scales};
use crate::{
    geometry::{Coordinates, mesh::Connectivity},
    math::{Quantity, Scalar, Tensor, TensorListVec},
    units::Area,
};
use std::sync::Mutex;

#[test]
fn parallel_evaluation_matches_the_serial_sum_exactly() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    assert!(hexes.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let theta = Quantity::new(0.3);
    let mut total = Quantity::<Area>::default();
    let mut gradient = Gradient::zero(coordinates.len());
    hexes.iter().zip(&scales).for_each(|(hex, &scale)| {
        let nodes = hex.iter().map(|&node| coordinates[node].clone()).collect();
        let (value, slopes) = terms(theta, scale, &nodes, true);
        total -= value;
        hex.iter()
            .zip(slopes)
            .for_each(|(&node, slope)| gradient[node] -= slope)
    });
    let energy = Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales,
        scratch: Mutex::default(),
        theta,
    };
    let mut parallel = Gradient::zero(coordinates.len());
    let value = energy.evaluate(&coordinates, Some(&mut parallel));
    assert_eq!(value.value().to_bits(), total.value().to_bits());
    gradient.iter().zip(parallel.iter()).for_each(|(a, b)| {
        (0..3).for_each(|i| assert_eq!(a[i].value().to_bits(), b[i].value().to_bits()))
    });
    assert_eq!(
        energy.evaluate(&coordinates, None).value().to_bits(),
        total.value().to_bits()
    );
}

#[test]
fn evaluation_keeps_nothing_from_the_previous_call() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    assert!(hexes.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let mut stretched = coordinates.clone();
    stretched.iter_mut().for_each(|node| node[0] *= 1.1);
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.3),
    };
    let gradient = |energy: &Energy, at: &Coordinates<3>| {
        let mut slopes = Gradient::zero(at.len());
        let value = energy.evaluate(at, Some(&mut slopes));
        (value.value().to_bits(), slopes)
    };
    let value = |energy: &Energy, at: &Coordinates<3>| energy.evaluate(at, None).value().to_bits();
    let (fresh_a, fresh_b) = (energy(), energy());
    let reference_a = gradient(&fresh_a, &coordinates);
    let reference_b = gradient(&fresh_b, &stretched);
    assert_ne!(reference_a.0, reference_b.0);
    let reused = energy();
    for (at, reference) in [
        (&coordinates, &reference_a),
        (&stretched, &reference_b),
        (&coordinates, &reference_a),
    ] {
        let (bits, slopes) = gradient(&reused, at);
        assert_eq!(bits, reference.0);
        slopes.iter().zip(reference.1.iter()).for_each(|(a, b)| {
            (0..3).for_each(|i| assert_eq!(a[i].value().to_bits(), b[i].value().to_bits()))
        });
    }
    assert_eq!(value(&reused, &stretched), reference_b.0);
    assert_eq!(value(&reused, &coordinates), reference_a.0);
    assert_eq!(value(&reused, &stretched), reference_b.0);
}

#[test]
fn skipping_hexes_that_cannot_reach_the_threshold_changes_nothing() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let coordinates = mesh.coordinates().clone();
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.3),
    };
    let reused = energy();
    reused.evaluate(&coordinates, Some(&mut Gradient::zero(coordinates.len())));
    for (round, amplitude) in [1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2]
        .into_iter()
        .enumerate()
    {
        let mut moved = coordinates.clone();
        moved.iter_mut().enumerate().for_each(|(n, node)| {
            (0..3).for_each(|i| {
                node[i] += Quantity::new(amplitude * offset(3 * n + i, round as Scalar))
            })
        });
        assert_eq!(
            reused.evaluate(&moved, None).value().to_bits(),
            energy().evaluate(&moved, None).value().to_bits(),
            "amplitude {amplitude}"
        );
    }
}

#[test]
fn skipping_hexes_in_the_gradient_changes_nothing_over_many_moves() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let mut coordinates = mesh.coordinates().clone();
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.6),
    };
    let gradient = |energy: &Energy, at: &Coordinates<3>| {
        let mut slopes = Gradient::zero(at.len());
        let value = energy.evaluate(at, Some(&mut slopes));
        (value.value().to_bits(), slopes)
    };
    let reused = energy();
    gradient(&reused, &coordinates);
    for round in 0..40 {
        coordinates.iter_mut().enumerate().for_each(|(n, node)| {
            let height = node[1];
            node[0] += height * 0.008;
            (0..3)
                .for_each(|i| node[i] += Quantity::new(1.0e-5 * offset(3 * n + i, round as Scalar)))
        });
        let (value, slopes) = gradient(&reused, &coordinates);
        let (fresh_value, fresh_slopes) = gradient(&energy(), &coordinates);
        assert_eq!(value, fresh_value, "value in round {round}");
        slopes.iter().zip(fresh_slopes.iter()).for_each(|(a, b)| {
            (0..3).for_each(|i| {
                assert_eq!(
                    a[i].value().to_bits(),
                    b[i].value().to_bits(),
                    "round {round}"
                )
            })
        });
        let mut trial = coordinates.clone();
        trial
            .iter_mut()
            .for_each(|node| node[0] += Quantity::new(1.0e-4));
        assert_eq!(
            reused.evaluate(&trial, None).value().to_bits(),
            energy().evaluate(&trial, None).value().to_bits(),
            "trial in round {round}"
        );
    }
}

#[test]
fn the_decayed_bound_never_exceeds_what_the_moved_hexahedron_still_has() {
    let nodes = distorted(0.15, 2.0);
    let (sj, _, _) = {
        let (_, _, (margin, shortest)) = bounded(
            &KINDS[0],
            Quantity::new(0.0),
            Quantity::new(1.0),
            &typed(&nodes),
            true,
        );
        (margin.value(), shortest.value(), ())
    };
    assert!(sj > 0.3);
    let theta = Quantity::new(sj - 0.2);
    let (_, _, bound) = bounded(&KINDS[0], theta, Quantity::new(1.0), &typed(&nodes), true);
    let base: Coordinates<3> = nodes.to_vec().into();
    let bounds = [bound];
    let hex = [0, 1, 2, 3, 4, 5, 6, 7];
    let (margin, shortest) = bound;
    let step = (margin - Quantity::new(1.0e-9)) * shortest / 12.0 * 0.9;
    let mut skipped = 0;
    for seed in 0..200 {
        let mut moved = nodes;
        moved.iter_mut().enumerate().for_each(|(n, node)| {
            (0..3).for_each(|i| {
                let sign = if offset(3 * n + i, seed as Scalar) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                node[i] += sign * step.value() / 3.0_f64.sqrt()
            })
        });
        let current: Coordinates<3> = moved.to_vec().into();
        let mut displaced = Vec::new();
        displacements(&current, &base, &mut displaced);
        let reach = Reach {
            bounds: &bounds,
            moved: &displaced,
        };
        let Some(distance) = reach.within(0, &hex) else {
            continue;
        };
        skipped += 1;
        let (lower, shorter) = reach.decayed(0, distance);
        let (_, _, (margin, shortest)) =
            bounded(&KINDS[0], theta, Quantity::new(1.0), &typed(&moved), true);
        assert!(margin >= lower && shortest >= shorter, "seed {seed}");
    }
    assert_eq!(skipped, 200);
}
