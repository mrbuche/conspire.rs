use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity, FiniteElementBlock,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                GradientVectors,
                linear::pyramid::{G, N, Pyramid},
                solid::SolidFiniteElement,
                test::test_finite_element,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
            test::test_finite_element_block,
        },
    },
    math::{ScalarList, Tensor, optimize::EqualityConstraint},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

const D: usize = 9;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [4, 5, 1, 0, 8],
        [5, 6, 2, 1, 8],
        [6, 7, 3, 2, 8],
        [0, 3, 7, 4, 8],
        [0, 1, 2, 3, 8],
        [5, 4, 7, 6, 8],
    ]
}

fn get_coordinates_block() -> NodalCoordinates {
    NodalCoordinates::from([
        [0.04175951, 0.00963520, -0.08547185],
        [1.08264022, 0.06657146, -0.06028449],
        [1.03545020, 0.95664729, 0.02444034],
        [0.03195872, 0.91151568, 0.01357932],
        [0.05957727, 0.09722483, 0.95352398],
        [1.09602809, 0.05991935, 0.92856463],
        [1.00712265, 0.99487330, 0.97093928],
        [0.03305756, 1.06846662, 1.02871468],
        [0.55951995, 0.55421498, 0.56169451],
    ])
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ])
}

fn get_velocities_block() -> NodalVelocities {
    NodalVelocities::from([
        [0.09637312, -0.01949176, -0.08003089],
        [-0.01054128, -0.08224619, 0.09007138],
        [-0.08801674, 0.01192077, -0.05977311],
        [-0.03106777, -0.08436228, 0.08698215],
        [0.08380139, 0.08364268, 0.07429425],
        [-0.08937229, 0.04991443, 0.02990664],
        [-0.02086137, 0.07165514, 0.02507809],
        [0.04689618, 0.07895952, -0.03680161],
        [-0.05106733, -0.02880547, -0.06480681],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    let strain = 0.55;
    let mut a = crate::math::Matrix::zero(11, 3 * D);
    a[0][3] = 1.0;
    a[1][6] = 1.0;
    a[2][15] = 1.0;
    a[3][18] = 1.0;
    a[4][0] = 1.0;
    a[5][9] = 1.0;
    a[6][12] = 1.0;
    a[7][21] = 1.0;
    a[8][1] = 1.0;
    a[9][2] = 1.0;
    a[10][11] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 1.0 + strain;
    b[1] = 1.0 + strain;
    b[2] = 1.0 + strain;
    b[3] = 1.0 + strain;
    b[4] = 0.0;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 0.0;
    b[8] = 0.0;
    b[9] = 0.0;
    b[10] = 0.0;
    (
        crate::constitutive::solid::elastic::AppliedLoad::UniaxialStress(strain + 1.0),
        a,
        b,
    )
}

fn applied_velocity(
    times: &crate::math::Vector,
) -> crate::constitutive::solid::viscoelastic::AppliedLoad<'_> {
    crate::constitutive::solid::viscoelastic::AppliedLoad::UniaxialStress(
        |_| 0.23,
        times.as_slice(),
    )
}

fn applied_velocities() -> (crate::math::Matrix, crate::math::Vector) {
    let velocity = 0.23;
    let mut a = crate::math::Matrix::zero(11, 3 * D);
    a[0][3] = 1.0;
    a[1][6] = 1.0;
    a[2][15] = 1.0;
    a[3][18] = 1.0;
    a[4][0] = 1.0;
    a[5][9] = 1.0;
    a[6][12] = 1.0;
    a[7][21] = 1.0;
    a[8][1] = 1.0;
    a[9][2] = 1.0;
    a[10][11] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = velocity;
    b[1] = velocity;
    b[2] = velocity;
    b[3] = velocity;
    b[4] = 0.0;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 0.0;
    b[8] = 0.0;
    b[9] = 0.0;
    b[10] = 0.0;
    (a, b)
}

test_finite_element!(Pyramid);
test_finite_element_block!(Pyramid);
