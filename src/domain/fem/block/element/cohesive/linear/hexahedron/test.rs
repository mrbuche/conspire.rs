use crate::{
    EPSILON,
    constitutive::cohesive::elastic::LinearElastic,
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
        cohesive::{
            elastic::ElasticCohesiveElement,
            linear::hexahedron::{Hexahedron, N, P},
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

const TANGENTIAL_TRACTION: Scalar = TANGENTIAL_STIFFNESS * TANGENTIAL_DISPLACEMENT;
const NORMAL_TRACTION: Scalar = NORMAL_STIFFNESS * NORMAL_DISPLACEMENT;

const COORDINATES: [[Scalar; 3]; N] = [
    [-0.47979299, 0.48230032, 0.0],
    [2.69165013, 0.37308724, 0.0],
    [2.69165013, 3.2116273, 0.0],
    [-0.47600989, 3.2116273, 0.0],
    [-0.47979299, 0.48230032, 0.0],
    [2.69165013, 0.37308724, 0.0],
    [2.69165013, 3.2116273, 0.0],
    [-0.47600989, 3.2116273, 0.0],
];
const MODEL: LinearElastic = LinearElastic {
    normal_stiffness: NORMAL_STIFFNESS,
    tangential_stiffness: TANGENTIAL_STIFFNESS,
};

#[test]
fn temporary_1() -> Result<(), TestError> {
    let coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    let element = Hexahedron::from(coordinates.clone());
    assert_eq_within_tols(
        &element.nodal_forces(&MODEL, &coordinates.into())?,
        &[[0.0; 3]; N].into(),
    )
}

#[test]
fn temporary_2() -> Result<(), TestError> {
    let mut coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    let element = Hexahedron::from(coordinates.clone());
    coordinates.iter_mut().skip(P).for_each(|coordinate| {
        coordinate[0] += TANGENTIAL_DISPLACEMENT;
        coordinate[2] += NORMAL_DISPLACEMENT
    });
    let area = element.integration_weights().into_iter().sum::<Scalar>();
    // Different than wedge since shape function gradients not constant.
    // Instead the total force is computed and compared to the traction.
    let forces = element.nodal_forces(&MODEL, &coordinates.into())?;
    assert_eq_within_tols(
        &forces.iter().take(P).map(|force| -force[0]).sum(),
        &(TANGENTIAL_TRACTION * area),
    )?;
    assert_eq_within_tols(
        &forces.iter().skip(P).map(|force| force[0]).sum(),
        &(TANGENTIAL_TRACTION * area),
    )?;
    forces
        .iter()
        .try_for_each(|force| assert_eq_within_tols(&force[1], &0.0))?;
    assert_eq_within_tols(
        &forces.iter().take(P).map(|force| -force[2]).sum(),
        &(NORMAL_TRACTION * area),
    )?;
    assert_eq_within_tols(
        &forces.iter().skip(P).map(|force| force[2]).sum(),
        &(NORMAL_TRACTION * area),
    )
}

#[test]
fn temporary_3() -> Result<(), TestError> {
    let coordinates = ElementNodalReferenceCoordinates::from(COORDINATES)
        .iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect::<ElementNodalReferenceCoordinates<N>>();
    let element = Hexahedron::from(coordinates.clone());
    assert_eq_within_tols(
        &element.nodal_forces(&MODEL, &coordinates.into())?,
        &[[0.0; 3]; N].into(),
    )
}

#[test]
fn temporary_4() -> Result<(), TestError> {
    let coordinates_0 = ElementNodalReferenceCoordinates::from(COORDINATES)
        .iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect::<ElementNodalReferenceCoordinates<N>>();
    let element = Hexahedron::from(coordinates_0);
    let mut coordinates = ElementNodalReferenceCoordinates::from(COORDINATES);
    coordinates.iter_mut().skip(P).for_each(|coordinate| {
        coordinate[0] += TANGENTIAL_DISPLACEMENT;
        coordinate[2] += NORMAL_DISPLACEMENT;
    });
    coordinates = coordinates
        .into_iter()
        .map(|coordinate| get_rotation_reference_configuration() * coordinate)
        .collect();
    let area = element.integration_weights().into_iter().sum::<Scalar>();
    // Different than wedge since shape function gradients not constant.
    // Instead the total force is computed and compared to the traction.
    let forces = element
        .nodal_forces(&MODEL, &coordinates.into())?
        .into_iter()
        .map(|nodal_force| {
            TensorRank2::<3, 1, 1>::from(get_rotation_reference_configuration().transpose())
                * nodal_force
        })
        .collect::<ElementNodalForcesSolid<N>>();
    assert_eq_within_tols(
        &forces.iter().take(P).map(|force| -force[0]).sum(),
        &(TANGENTIAL_TRACTION * area),
    )?;
    assert_eq_within_tols(
        &forces.iter().skip(P).map(|force| force[0]).sum(),
        &(TANGENTIAL_TRACTION * area),
    )?;
    forces
        .iter()
        .try_for_each(|force| assert_eq_within_tols(&force[1], &0.0))?;
    assert_eq_within_tols(
        &forces.iter().take(P).map(|force| -force[2]).sum(),
        &(NORMAL_TRACTION * area),
    )?;
    assert_eq_within_tols(
        &forces.iter().skip(P).map(|force| force[2]).sum(),
        &(NORMAL_TRACTION * area),
    )
}

#[test]
fn temporary_5() -> Result<(), TestError> {
    let coordinates_0 = ElementNodalReferenceCoordinates::from(COORDINATES);
    let coordinates = ElementNodalCoordinates::from(coordinates_0.clone());
    let element = Hexahedron::from(coordinates_0);
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
                                        element.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    nodal_coordinates[b][j] -= EPSILON;
                                    finite_difference -=
                                        element.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
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
        &element.nodal_stiffnesses(&MODEL, &coordinates)?,
        &nodal_stiffnesses_fd,
    )
}

#[test]
fn temporary_6() -> Result<(), TestError> {
    let coordinates_0 = ElementNodalReferenceCoordinates::from([
        [-0.57177033, -0.20395894, 0.23629102],
        [1.49477913, 1.72253902, 1.40527015],
        [1.49477913, -0.2546453, 1.40527015],
        [-2.31789525, -0.2546453, 2.40281722],
        [-0.57177033, -0.20395894, 0.23629102],
        [1.49477913, 1.72253902, 1.40527015],
        [1.49477913, -0.2546453, 1.40527015],
        [-2.31789525, -0.2546453, 2.40281722],
    ]);
    let element = Hexahedron::from(coordinates_0);
    let coordinates = ElementNodalCoordinates::from([
        [-0.64542355, -0.31521986, 0.2103109],
        [1.50161765, 1.80846799, 1.49664724],
        [1.50161765, -0.08562506, 1.49664724],
        [-2.29750971, -0.08562506, 2.28063606],
        [4.72044386, 3.95736046, 4.01544368],
        [6.80745386, 6.13361434, 5.46225216],
        [6.80745386, 3.98543986, 5.46225216],
        [3.14323173, 3.98543986, 6.22717385],
    ]);
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
                                        element.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
                                    nodal_coordinates[b][j] -= EPSILON;
                                    finite_difference -=
                                        element.nodal_forces(&MODEL, &nodal_coordinates)?[a][i];
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
        &element.nodal_stiffnesses(&MODEL, &coordinates)?,
        &nodal_stiffnesses_fd,
    )
}
