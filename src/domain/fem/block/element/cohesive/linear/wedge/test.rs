use crate::{
    constitutive::cohesive::elastic::LinearElastic,
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement,
        cohesive::{
            elastic::ElasticCohesiveElement,
            linear::wedge::{N, P, Wedge},
        },
    },
    math::{Scalar, Tensor, assert_eq_within_tols, test::TestError},
    mechanics::test::get_rotation_reference_configuration,
};

const NORMAL_DISPLACEMENT: Scalar = 1.2;
const NORMAL_STIFFNESS: Scalar = 3.4;
const TANGENTIAL_STIFFNESS: Scalar = 5.6;
const TANGENTIAL_DISPLACEMENT: Scalar = 7.8;

const TANGENTIAL_TRACTION_P: Scalar = TANGENTIAL_STIFFNESS * TANGENTIAL_DISPLACEMENT / P as Scalar;
const NORMAL_TRACTION_P: Scalar = NORMAL_STIFFNESS * NORMAL_DISPLACEMENT / P as Scalar;

const COORDINATES: [[Scalar; 3]; N] = [
    [-0.47979299, 0.48230032, 0.0],
    [2.69165013, 0.37308724, 0.0],
    [-0.47600989, 3.2116273, 0.0],
    [-0.47979299, 0.48230032, 0.0],
    [2.69165013, 0.37308724, 0.0],
    [-0.47600989, 3.2116273, 0.0],
];
const MODEL: LinearElastic = LinearElastic {
    normal_stiffness: NORMAL_STIFFNESS,
    tangential_stiffness: TANGENTIAL_STIFFNESS,
};

#[test]
fn temporary_1() -> Result<(), TestError> {
    let coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    let wedge = Wedge::from(coordinates.clone());
    assert_eq_within_tols(
        &wedge.nodal_forces(&MODEL, &coordinates.into())?,
        &[[0.0; 3]; N].into(),
    )
}

#[test]
fn temporary_2() -> Result<(), TestError> {
    let mut coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    let wedge = Wedge::from(coordinates.clone());
    coordinates.iter_mut().skip(P).for_each(|coordinate| {
        coordinate[0] += TANGENTIAL_DISPLACEMENT;
        coordinate[2] += NORMAL_DISPLACEMENT
    });
    let area = wedge.integration_weights().into_iter().sum::<Scalar>();
    let tangential_force = TANGENTIAL_TRACTION_P * area;
    let normal_force = NORMAL_TRACTION_P * area;

println!("{:?}", wedge.nodal_forces(&MODEL, &coordinates.clone().into())?);
println!("{tangential_force}, {normal_force}");

    assert_eq_within_tols(
        &wedge.nodal_forces(&MODEL, &coordinates.into())?,
        &[
            [-tangential_force, 0.0, normal_force],
            [-tangential_force, 0.0, normal_force],
            [-tangential_force, 0.0, normal_force],
            [-tangential_force, 0.0, -normal_force],
            [-tangential_force, 0.0, -normal_force],
            [-tangential_force, 0.0, -normal_force],
        ]
        .into(),
    )
}

#[test]
fn temporary_3() -> Result<(), TestError> {
    let coordinates = ElementNodalReferenceCoordinates::from(COORDINATES)
        .iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect::<ElementNodalReferenceCoordinates<N>>();
    let wedge = Wedge::from(coordinates.clone());
    assert_eq_within_tols(
        &wedge.nodal_forces(&MODEL, &coordinates.into())?,
        &[[0.0; 3]; N].into(),
    )
}

#[test]
fn temporary_4() -> Result<(), TestError> {
    todo!("rotate normal/shear")
}
