use crate::{
    cbm::{Model, NodalReferenceCoordinates, block::Block},
    constitutive::{electrical::dielectric::Linear, thermal::conduction::Fourier},
    domain::{
        diffusion::DiffusionElements, electrical::NodalPotentials, thermal::NodalTemperatures,
    },
    geometry::mesh::PrimitiveConnectivity,
    math::{Hessian, Tensor},
    units::{ElectricPotential, Permittivity, PowerPerLengthTemperature, Temperature},
};

const NODES: usize = 5;

const COORDINATES: [[f64; 3]; NODES] = [
    [0.1, 0.2, 0.0],
    [1.3, 0.1, 0.2],
    [0.2, 0.9, 0.1],
    [0.3, 0.4, 1.2],
    [1.5, 1.4, 1.3],
];

const VOLTS: [f64; NODES] = [3.0, 11.0, -4.0, 7.5, 0.5];

fn reference() -> NodalReferenceCoordinates<3> {
    NodalReferenceCoordinates::from(COORDINATES)
}

fn connectivity() -> PrimitiveConnectivity<3, 4> {
    PrimitiveConnectivity::from(vec![[0, 1, 2, 3], [1, 2, 3, 4]])
}

fn model() -> Model<Block<Linear>, 3> {
    (
        Block::<Linear>::from((
            Linear {
                permittivity: Permittivity::farads_per_meter(2.5),
            },
            connectivity(),
            &reference(),
        )),
        reference(),
    )
        .into()
}

fn thermal_model() -> Model<Block<Fourier>, 3> {
    (
        Block::<Fourier>::from((
            Fourier {
                thermal_conductivity: PowerPerLengthTemperature::watts_per_meter_kelvin(2.5),
            },
            connectivity(),
            &reference(),
        )),
        reference(),
    )
        .into()
}

fn potentials(volts: [f64; NODES]) -> NodalPotentials {
    volts.map(ElectricPotential::volts).into()
}

fn near(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{a} is not within {tolerance} of {b}"
    );
}

#[test]
fn the_charge_is_the_potential_gradient() {
    let model = model();
    let initial = potentials(VOLTS);
    let charges = model.nodal_forces(&initial).unwrap();
    let step = 1e-4;
    (0..NODES).for_each(|node| {
        let shifted = |sign: f64| {
            let mut shifted = initial.clone();
            shifted[node] += ElectricPotential::volts(sign * step);
            model.potential(&shifted).unwrap().value()
        };
        near(
            (shifted(1.0) - shifted(-1.0)) / (2.0 * step),
            charges[node].value(),
            1e-6,
        )
    });
}

#[test]
fn the_stiffness_is_the_symmetric_derivative_of_the_charge() {
    let model = model();
    let initial = potentials(VOLTS);
    let stiffnesses = model.nodal_stiffnesses(&initial).unwrap();
    let step = 1.0;
    (0..NODES).for_each(|column| {
        let shifted = |sign: f64| {
            let mut shifted = initial.clone();
            shifted[column] += ElectricPotential::volts(sign * step);
            model.nodal_forces(&shifted).unwrap()
        };
        let (up, down) = (shifted(1.0), shifted(-1.0));
        (0..NODES).for_each(|row| {
            near(
                (up[row].value() - down[row].value()) / (2.0 * step),
                stiffnesses.entry(row, column),
                1e-8,
            );
            near(
                stiffnesses.entry(row, column),
                stiffnesses.entry(column, row),
                1e-12,
            )
        })
    });
}

#[test]
fn charge_is_conserved_and_a_uniform_potential_has_none() {
    let model = model();
    let total: f64 = model
        .nodal_forces(&potentials(VOLTS))
        .unwrap()
        .iter()
        .map(|charge| charge.value())
        .sum();
    near(total, 0.0, 1e-12);
    model
        .nodal_forces(&potentials([7.3; NODES]))
        .unwrap()
        .iter()
        .for_each(|charge| near(charge.value(), 0.0, 1e-12));
}

#[test]
fn the_charges_match_the_heat_flows_of_a_conductor_with_the_same_number() {
    let charges = model().nodal_forces(&potentials(VOLTS)).unwrap();
    let flows = thermal_model()
        .nodal_forces(&NodalTemperatures::from(VOLTS.map(Temperature::kelvin)))
        .unwrap();
    (0..NODES).for_each(|node| near(charges[node].value(), flows[node].value(), 1e-12));
}
