use crate::{
    cbm::{
        ElasticExplicitDynamics, Execution, Lumped, Model, NodalCoordinates,
        NodalReferenceCoordinates, block::Block,
    },
    constitutive::solid::hyperelastic::NeoHookean,
    domain::{solid::dynamics::test::fixed_dofs, stability::Unchecked},
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Stress, Time},
};
use core::array::from_fn;

const LAYERS: usize = 40;

type B = Block<NeoHookean, Quantity<Density>>;

fn node(i: usize, j: usize, k: usize) -> usize {
    4 * k + 2 * j + i
}

fn coordinates() -> Vec<[f64; 3]> {
    (0..=LAYERS)
        .flat_map(|k| (0..2).flat_map(move |j| (0..2).map(move |i| [i as f64, j as f64, k as f64])))
        .collect()
}

fn connectivity() -> Vec<[usize; 4]> {
    let coordinates = coordinates();
    let volume = |t: &[usize; 4]| {
        let edge = |a: usize| from_fn::<f64, 3, _>(|c| coordinates[t[a]][c] - coordinates[t[0]][c]);
        let (a, b, c) = (edge(1), edge(2), edge(3));
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    (0..LAYERS)
        .flat_map(|k| {
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ]
            .map(|order| {
                let mut at = [0, 0, k];
                let mut tetrahedron = [node(0, 0, k), 0, 0, node(1, 1, k + 1)];
                order.iter().enumerate().for_each(|(step, &axis)| {
                    at[axis] += 1;
                    tetrahedron[step + 1] = node(at[0], at[1], at[2]);
                });
                tetrahedron[3] = node(1, 1, k + 1);
                if volume(&tetrahedron) < 0.0 {
                    tetrahedron.swap(1, 2);
                }
                tetrahedron
            })
        })
        .collect()
}

fn model() -> Model<B, 3> {
    let reference = NodalReferenceCoordinates::from(coordinates());
    (
        (
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            Density::kilograms_per_cubic_meter(7.8e3),
            connectivity().into(),
            &reference,
        )
            .into(),
        reference,
    )
        .into()
}

fn run(model: &Model<B, 3>, execution: Execution) -> (NodalCoordinates<3>, Vec<[f64; 3]>) {
    let mut stretched = coordinates();
    let top = stretched.len() - 1;
    stretched[top][0] += 0.1;
    stretched[top / 2][1] -= 0.05;
    let (_, coordinates, _, accelerations) = model
        .integrate(
            execution,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(1e-7)),
            &[Time::seconds(0.0), Time::seconds(2e-6)],
            (stretched.into(), vec![[0.0; 3]; top + 1].into()),
            &(),
            &fixed_dofs(&[]),
        )
        .unwrap();
    (
        coordinates[coordinates.len() - 1].clone(),
        accelerations[accelerations.len() - 1]
            .iter()
            .map(|acceleration| acceleration.iter().map(|a| a.value()).collect::<Vec<_>>())
            .map(|a| [a[0], a[1], a[2]])
            .collect(),
    )
}

#[test]
fn the_body_is_forced_by_the_stretch() {
    let (_, accelerations) = run(&model(), Execution::Serial);
    assert!(accelerations.iter().flatten().all(|a| a.is_finite()));
    assert!(accelerations.iter().flatten().any(|&a| a != 0.0));
}

#[test]
fn any_number_of_threads_gives_the_serial_run() {
    let model = model();
    let serial = run(&model, Execution::Serial);
    [1, 2, 3, 7, 0].iter().for_each(|&threads| {
        let parallel = run(&model, Execution::Threads(threads));
        assert!(parallel.0 == serial.0, "{threads} threads");
        assert!(parallel.1 == serial.1, "{threads} threads");
    });
}
