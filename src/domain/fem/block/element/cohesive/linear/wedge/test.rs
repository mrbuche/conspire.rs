use crate::{
    EPSILON,
    constitutive::cohesive::elastic::LinearElastic,
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
        cohesive::{
            elastic::ElasticCohesiveElement,
            linear::wedge::{N, P, Wedge},
        },
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
    math::{
        Rank2, Scalar, Tensor, TensorRank2, assert_eq_within_tols,
        test::{TestError, assert_eq_from_fd},
    },
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
    assert_eq_within_tols(
        &wedge.nodal_forces(&MODEL, &coordinates.into())?,
        &[
            [tangential_force, 0.0, normal_force],
            [tangential_force, 0.0, normal_force],
            [tangential_force, 0.0, normal_force],
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
    let coordinates_0 = ElementNodalReferenceCoordinates::from(COORDINATES)
        .iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect::<ElementNodalReferenceCoordinates<N>>();
    let wedge = Wedge::from(coordinates_0);
    let mut coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    coordinates.iter_mut().skip(P).for_each(|coordinate| {
        coordinate[0] += TANGENTIAL_DISPLACEMENT;
        coordinate[2] += NORMAL_DISPLACEMENT;
    });
    coordinates = coordinates
        .into_iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect();
    let area = wedge.integration_weights().into_iter().sum::<Scalar>();
    let tangential_force = TANGENTIAL_TRACTION_P * area;
    let normal_force = NORMAL_TRACTION_P * area;
    let nodal_forces_rotated_back = wedge
        .nodal_forces(&MODEL, &coordinates.into())?
        .into_iter()
        .map(|nodal_force| {
            TensorRank2::<3, 1, 1>::from(get_rotation_reference_configuration().transpose())
                * nodal_force
        })
        .collect::<ElementNodalForcesSolid<N>>();
    assert_eq_within_tols(
        &nodal_forces_rotated_back,
        &[
            [tangential_force, 0.0, normal_force],
            [tangential_force, 0.0, normal_force],
            [tangential_force, 0.0, normal_force],
            [-tangential_force, 0.0, -normal_force],
            [-tangential_force, 0.0, -normal_force],
            [-tangential_force, 0.0, -normal_force],
        ]
        .into(),
    )
}

#[test]
fn temporary_5() -> Result<(), TestError> {
    let coordinates_0 = ElementNodalReferenceCoordinates::from(COORDINATES);
    let coordinates = ElementNodalCoordinates::from(coordinates_0.clone());
    let wedge = Wedge::from(coordinates_0);
    let mut finite_difference = 0.0;
    let nodal_stiffnesses_fd = (0..N)
        .map(|a| {
            (0..N)
                .map(|b| {
                    (0..3)
                        .map(|i| {
                            (0..3)
                                .map(|j| {
                                    let mut nodal_coordinates = coordinates.clone();
                                    nodal_coordinates[b][j] += 0.5 * EPSILON;
                                    finite_difference =
                                        wedge.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    nodal_coordinates[b][j] -= EPSILON;
                                    finite_difference -=
                                        wedge.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    Ok(finite_difference / EPSILON)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect::<Result<ElementNodalStiffnessesSolid<N>, TestError>>()?;
    assert_eq_from_fd(
        &wedge.nodal_stiffnesses(&MODEL, &coordinates)?,
        &nodal_stiffnesses_fd,
    )
}

#[test]
fn temporary_6() -> Result<(), TestError> {
    let coordinates_0 = ElementNodalReferenceCoordinates::from([
        [-0.57177033, -0.20395894, 0.23629102],
        [1.49477913, 1.72253902, 1.40527015],
        [-2.31789525, -0.2546453, 2.40281722],
        [-0.57177033, -0.20395894, 0.23629102],
        [1.49477913, 1.72253902, 1.40527015],
        [-2.31789525, -0.2546453, 2.40281722],
    ]);
    let wedge = Wedge::from(coordinates_0);
    let coordinates = ElementNodalCoordinates::from([
        [-0.589159, -0.293693, 0.171611],
        [1.419824, 1.79346, 1.466893],
        [-2.258207, -0.161917, 2.327105],
        [4.787346, 3.952276, 4.088873],
        [6.796329, 6.039429, 5.384155],
        [3.118298, 4.084052, 6.244367],
    ]);
    //
    // Add noise to coordinates after passes without it!
    //
    let mut finite_difference = 0.0;
    let nodal_stiffnesses_fd = (0..N)
        .map(|a| {
            (0..N)
                .map(|b| {
                    (0..3)
                        .map(|i| {
                            (0..3)
                                .map(|j| {
                                    let mut nodal_coordinates = coordinates.clone();
                                    nodal_coordinates[b][j] += 0.5 * EPSILON;
                                    finite_difference =
                                        wedge.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    nodal_coordinates[b][j] -= EPSILON;
                                    finite_difference -=
                                        wedge.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    Ok(finite_difference / EPSILON)
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect::<Result<ElementNodalStiffnessesSolid<N>, TestError>>()?;
    assert_eq_from_fd(
        &wedge.nodal_stiffnesses(&MODEL, &coordinates)?,
        &nodal_stiffnesses_fd,
    )
}
