use crate::{
    fem::{
        Connectivity, ElementBlock, GradientVectors, NodalCoordinates, NodalCoordinatesBlock,
        NodalForcesBlock, NodalStiffnessesBlock, NodalVelocities, NodalVelocitiesBlock,
        ReferenceNodalCoordinates, ReferenceNodalCoordinatesBlock,
        block::{
            FiniteElementBlock,
            element::{
                FiniteElement, FiniteElementMethods,
                linear::tetrahedron::{G, N, Tetrahedron},
                test::test_finite_element,
            },
            test::test_finite_element_block,
        },
    },
    math::{Scalars, Tensor, optimize::EqualityConstraint},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

const D: usize = 14;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [13, 12, 8, 1],
        [10, 3, 0, 8],
        [11, 10, 8, 3],
        [12, 11, 8, 2],
        [11, 2, 3, 8],
        [12, 2, 8, 1],
        [13, 10, 5, 0],
        [13, 11, 10, 8],
        [10, 6, 9, 5],
        [12, 7, 4, 9],
        [12, 11, 7, 9],
        [11, 7, 9, 6],
        [13, 1, 8, 0],
        [13, 9, 4, 5],
        [13, 12, 1, 4],
        [11, 10, 6, 9],
        [11, 10, 3, 6],
        [12, 11, 2, 7],
        [13, 11, 9, 10],
        [13, 12, 4, 9],
        [13, 10, 0, 8],
        [13, 10, 9, 5],
        [13, 12, 11, 8],
        [13, 12, 9, 11],
    ]
}

fn get_coordinates_block() -> NodalCoordinatesBlock {
    NodalCoordinatesBlock::from([
        [0.48419081, -0.52698494, 0.42026988],
        [0.43559430, 0.52696224, 0.54477963],
        [-0.56594965, 0.57076191, 0.51683869],
        [-0.56061746, -0.42795457, 0.55275658],
        [0.41878700, 0.53190268, -0.44744274],
        [0.47232357, -0.57252738, -0.42946606],
        [-0.45168197, -0.5102938, -0.57959825],
        [-0.41776733, 0.41581785, -0.45911886],
        [0.05946988, 0.03773822, 0.44149305],
        [-0.08478334, -0.09009810, -0.46105872],
        [-0.04039882, -0.58201398, 0.09346960],
        [-0.57820738, 0.08325131, 0.03614415],
        [-0.04145077, 0.56406301, 0.09988905],
        [0.52149656, -0.08553510, -0.03187069],
    ])
}

fn reference_coordinates() -> ReferenceNodalCoordinates<N> {
    ReferenceNodalCoordinates::new([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

fn get_reference_coordinates_block() -> ReferenceNodalCoordinatesBlock {
    ReferenceNodalCoordinatesBlock::from([
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, -0.5],
        [0.0, -0.5, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ])
}

fn get_velocities_block() -> NodalVelocitiesBlock {
    NodalVelocitiesBlock::from([
        [0.00888030, -0.09877116, 0.07861759],
        [0.02037718, -0.09870374, -0.04739945],
        [-0.02023814, -0.00392495, 0.00612573],
        [0.08198906, 0.09420134, -0.05701550],
        [-0.05278682, 0.02357548, 0.03048997],
        [-0.06860257, -0.08783628, -0.07055701],
        [-0.08624215, -0.04538965, -0.02892557],
        [-0.09304190, -0.07169055, -0.04272249],
        [0.04056852, -0.09734596, 0.00339223],
        [-0.08708972, -0.08251380, -0.08124456],
        [-0.03744580, -0.06003551, 0.09364016],
        [-0.06954597, 0.06645925, -0.08261904],
        [0.07740919, -0.00642660, 0.01101806],
        [-0.04079346, -0.07283644, 0.05569305],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    let strain = 0.55;
    let mut a = crate::math::Matrix::zero(13, 42);
    a[0][0] = 1.0;
    a[1][3] = 1.0;
    a[2][12] = 1.0;
    a[3][15] = 1.0;
    a[4][39] = 1.0;
    a[5][6] = 1.0;
    a[6][9] = 1.0;
    a[7][18] = 1.0;
    a[8][21] = 1.0;
    a[9][33] = 1.0;
    a[10][19] = 1.0;
    a[11][20] = 1.0;
    a[12][23] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 0.5 + strain;
    b[1] = 0.5 + strain;
    b[2] = 0.5 + strain;
    b[3] = 0.5 + strain;
    b[4] = 0.5 + strain;
    b[5] = -0.5;
    b[6] = -0.5;
    b[7] = -0.5;
    b[8] = -0.5;
    b[9] = -0.5;
    b[10] = -0.5;
    b[11] = -0.5;
    b[12] = -0.5;
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
    let mut a = crate::math::Matrix::zero(13, 42);
    a[0][0] = 1.0;
    a[1][3] = 1.0;
    a[2][12] = 1.0;
    a[3][15] = 1.0;
    a[4][39] = 1.0;
    a[5][6] = 1.0;
    a[6][9] = 1.0;
    a[7][18] = 1.0;
    a[8][21] = 1.0;
    a[9][33] = 1.0;
    a[10][19] = 1.0;
    a[11][20] = 1.0;
    a[12][23] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = velocity;
    b[1] = velocity;
    b[2] = velocity;
    b[3] = velocity;
    b[4] = velocity;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 0.0;
    b[8] = 0.0;
    b[9] = 0.0;
    b[10] = 0.0;
    b[11] = 0.0;
    b[12] = 0.0;
    (a, b)
}

test_finite_element!(Tetrahedron);
test_finite_element_block!(Tetrahedron);
