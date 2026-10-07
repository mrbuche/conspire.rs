use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        self, NodalCoordinates, NodalReferenceCoordinates, block::element::linear::Tetrahedron,
        time_scale::TimeScaleElements,
    },
    math::Quantity,
    units::{Density, Stress},
    vem::block::{Block, element::Element},
};

const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);

fn model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(1.0e9),
        shear_modulus: Stress::pascals(5.0e8),
    }
}

fn virtual_time_scale(nodes: &[[f64; 3]], faces: Vec<Vec<usize>>) -> f64 {
    let reference = NodalReferenceCoordinates::from(nodes.to_vec());
    Block::<_, Element>::from((model(), vec![(0..faces.len()).collect()], faces, &reference))
        .with_density(DENSITY)
        .fastest_time_scale(&reference, &NodalCoordinates::from(nodes.to_vec()))
        .unwrap()
        .value()
}

fn finite_time_scale(nodes: &[[f64; 3]], tetrahedra: Vec<[usize; 4]>) -> f64 {
    let reference = NodalReferenceCoordinates::from(nodes.to_vec());
    fem::block::Block::<_, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>::from((
        model(),
        DENSITY,
        tetrahedra,
        &reference,
    ))
    .fastest_time_scale(&reference, &NodalCoordinates::from(nodes.to_vec()))
    .unwrap()
    .value()
}

fn wedge(epsilon: f64) -> [[f64; 3]; 5] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.3, epsilon],
        [0.3, 0.3, -1.0],
    ]
}

fn wedge_faces() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 3],
        vec![1, 2, 3],
        vec![2, 0, 3],
        vec![1, 0, 4],
        vec![2, 1, 4],
        vec![0, 2, 4],
    ]
}

#[test]
fn a_tetrahedron_has_the_time_scale_of_the_finite_element() {
    let nodes = [
        [0.1, 0.2, 0.0],
        [1.3, 0.1, 0.2],
        [0.2, 0.9, 0.1],
        [0.3, 0.4, 1.2],
    ];
    let virtual_element = virtual_time_scale(
        &nodes,
        vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]],
    );
    let finite_element = finite_time_scale(&nodes, vec![[0, 1, 2, 3]]);
    assert!(
        (virtual_element / finite_element - 1.0).abs() < 1e-6,
        "{virtual_element} != {finite_element}"
    );
}

#[test]
fn an_agglomerated_wedge_keeps_its_time_scale_as_the_wedge_flattens() {
    let [mild, flat]: [f64; 2] =
        [1e-1, 1e-5].map(|epsilon| virtual_time_scale(&wedge(epsilon), wedge_faces()));
    assert!(mild / flat < 3.0 && flat / mild < 3.0, "{mild} vs {flat}");
}

#[test]
fn agglomerating_a_flat_wedge_beats_the_finite_elements() {
    let nodes = wedge(1e-5);
    let virtual_element = virtual_time_scale(&nodes, wedge_faces());
    let finite_elements = finite_time_scale(&nodes, vec![[0, 1, 2, 3], [0, 2, 1, 4]]);
    assert!(
        virtual_element / finite_elements > 1e3,
        "{virtual_element} vs {finite_elements}"
    );
}
