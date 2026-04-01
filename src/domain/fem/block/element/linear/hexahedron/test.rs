use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                FiniteElement, GradientVectors,
                linear::hexahedron::{G, Hexahedron, M, N, P},
                solid::SolidFiniteElement,
                test::test_finite_element,
            },
            test::test_finite_element_block,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{ScalarList, Tensor, optimize::EqualityConstraint},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

const D: usize = 27;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [0, 1, 4, 3, 9, 10, 13, 12],
        [1, 2, 5, 4, 10, 11, 14, 13],
        [3, 4, 7, 6, 12, 13, 16, 15],
        [4, 5, 8, 7, 13, 14, 17, 16],
        [9, 10, 13, 12, 18, 19, 22, 21],
        [10, 11, 14, 13, 19, 20, 23, 22],
        [12, 13, 16, 15, 21, 22, 25, 24],
        [13, 14, 17, 16, 22, 23, 26, 25],
    ]
}

fn get_coordinates_block() -> NodalCoordinates {
    NodalCoordinates::from([
        [-0.49344606, -0.44672548, -0.55524455],
        [0.00224851, -0.43237921, -0.46885579],
        [0.59974619, -0.58745455, -0.54662781],
        [-0.52981819, 0.09461814, -0.42502779],
        [-0.01262477, 0.05690753, -0.57021839],
        [0.59772588, 0.06366829, -0.58420224],
        [-0.55459006, 0.53748865, -0.40991277],
        [0.03674893, 0.5406329, -0.5403517],
        [0.50490308, 0.48314229, -0.43801276],
        [-0.46330299, -0.4557396, 0.06033268],
        [0.07039472, -0.54868351, -0.07254376],
        [0.48496913, -0.585256, 0.07111521],
        [-0.49508808, 0.08298562, 0.03886799],
        [-0.09874559, -0.01819436, 0.05979098],
        [0.50143268, -0.0228508, 0.07207578],
        [-0.42018895, 0.57832872, 0.02940452],
        [-0.0198192, 0.51915371, -0.01962645],
        [0.58928423, 0.52437576, 0.0813594],
        [-0.54517793, -0.57219519, 0.58668863],
        [0.06058873, -0.5360219, 0.48159493],
        [0.4256927, -0.50104237, 0.52503988],
        [-0.56030072, 0.060492, 0.51402389],
        [-0.02092666, 0.00430224, 0.4098474],
        [0.57071401, 0.05166412, 0.45854914],
        [-0.5825826, 0.47432363, 0.55823452],
        [0.02164257, 0.52156089, 0.48985245],
        [0.45546198, 0.44538201, 0.40452406],
    ])
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [-0.5, -0.5, -0.5],
        [0.0, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.0, -0.5],
        [0.0, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.5, -0.5],
        [0.0, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.0],
        [0.0, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.5],
        [0.0, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [-0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
}

fn get_velocities_block() -> NodalVelocities {
    NodalVelocities::from([
        [0.04705949, 0.03631753, 0.07485168],
        [-0.05308881, 0.00788418, -0.00509899],
        [0.09843214, 0.01365840, 0.03718527],
        [-0.03283459, -0.06107903, 0.03729790],
        [-0.06881257, 0.09663691, 0.05662889],
        [-0.09618856, 0.04674098, -0.07638976],
        [-0.04904809, 0.06095278, 0.06916505],
        [-0.05792408, -0.04024974, -0.07464079],
        [-0.08427349, -0.09329782, -0.03069372],
        [0.09699865, 0.08993480, 0.05100433],
        [0.06544895, -0.09933706, 0.04667646],
        [-0.03351784, -0.06347749, 0.01146035],
        [-0.01581637, 0.01924530, -0.01647996],
        [-0.07755037, 0.06224361, 0.04342054],
        [-0.00087397, 0.02165712, 0.03513907],
        [-0.09812976, -0.09209075, -0.00681481],
        [-0.09992033, -0.02172153, -0.02055542],
        [-0.04294789, -0.03508023, -0.01523334],
        [-0.01289026, -0.0787843, 0.04242054],
        [0.06040533, -0.00306949, -0.06926722],
        [0.05451315, -0.07387627, 0.08822557],
        [0.03834922, 0.03471130, 0.04046410],
        [-0.02460383, -0.06098678, -0.08874120],
        [-0.06386626, 0.09120429, 0.06585182],
        [-0.09463749, -0.0138323, 0.06690724],
        [-0.04392108, -0.00704832, 0.00025807],
        [-0.02136258, 0.03102960, -0.04881753],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    let strain = 0.55;
    let mut a = crate::math::Matrix::zero(21, 3 * D);
    a[0][6] = 1.0;
    a[1][15] = 1.0;
    a[2][24] = 1.0;
    a[3][33] = 1.0;
    a[4][42] = 1.0;
    a[5][51] = 1.0;
    a[6][60] = 1.0;
    a[7][69] = 1.0;
    a[8][78] = 1.0;
    a[9][0] = 1.0;
    a[10][9] = 1.0;
    a[11][18] = 1.0;
    a[12][27] = 1.0;
    a[13][36] = 1.0;
    a[14][45] = 1.0;
    a[15][54] = 1.0;
    a[16][63] = 1.0;
    a[17][72] = 1.0;
    a[18][1] = 1.0;
    a[19][2] = 1.0;
    a[20][20] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 0.5 + strain;
    b[1] = 0.5 + strain;
    b[2] = 0.5 + strain;
    b[3] = 0.5 + strain;
    b[4] = 0.5 + strain;
    b[5] = 0.5 + strain;
    b[6] = 0.5 + strain;
    b[7] = 0.5 + strain;
    b[8] = 0.5 + strain;
    b[9] = -0.5;
    b[10] = -0.5;
    b[11] = -0.5;
    b[12] = -0.5;
    b[13] = -0.5;
    b[14] = -0.5;
    b[15] = -0.5;
    b[16] = -0.5;
    b[17] = -0.5;
    b[18] = -0.5;
    b[19] = -0.5;
    b[20] = -0.5;
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
    let mut a = crate::math::Matrix::zero(21, 3 * D);
    a[0][6] = 1.0;
    a[1][15] = 1.0;
    a[2][24] = 1.0;
    a[3][33] = 1.0;
    a[4][42] = 1.0;
    a[5][51] = 1.0;
    a[6][60] = 1.0;
    a[7][69] = 1.0;
    a[8][78] = 1.0;
    a[9][0] = 1.0;
    a[10][9] = 1.0;
    a[11][18] = 1.0;
    a[12][27] = 1.0;
    a[13][36] = 1.0;
    a[14][45] = 1.0;
    a[15][54] = 1.0;
    a[16][63] = 1.0;
    a[17][72] = 1.0;
    a[18][1] = 1.0;
    a[19][2] = 1.0;
    a[20][20] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = velocity;
    b[1] = velocity;
    b[2] = velocity;
    b[3] = velocity;
    b[4] = velocity;
    b[5] = velocity;
    b[6] = velocity;
    b[7] = velocity;
    b[8] = velocity;
    b[9] = 0.0;
    b[10] = 0.0;
    b[11] = 0.0;
    b[12] = 0.0;
    b[13] = 0.0;
    b[14] = 0.0;
    b[15] = 0.0;
    b[16] = 0.0;
    b[17] = 0.0;
    b[18] = 0.0;
    b[19] = 0.0;
    b[20] = 0.0;
    (a, b)
}

test_finite_element!(Hexahedron);
test_finite_element_block!(Hexahedron);

mod minimum_scaled_jacobian {
    use super::*;
    use crate::math::test::{TestError, assert_eq_within_tols};
    #[test]
    fn ideal() -> Result<(), TestError> {
        let msj = Hexahedron::minimum_scaled_jacobian(reference_coordinates());
        assert_eq_within_tols(&msj, &1.0)
    }
    #[test]
    fn flat() -> Result<(), TestError> {
        let nodal_coordinates = reference_coordinates()
            .into_iter()
            .take(4)
            .chain(reference_coordinates().into_iter().take(4))
            .collect();
        let msj = Hexahedron::minimum_scaled_jacobian(nodal_coordinates);
        assert_eq_within_tols(&msj, &0.0)
    }
    #[test]
    fn inverted_ideal() -> Result<(), TestError> {
        let nodal_coordinates = reference_coordinates()
            .into_iter()
            .skip(4)
            .chain(reference_coordinates().into_iter().take(4))
            .collect();
        let msj = Hexahedron::minimum_scaled_jacobian(nodal_coordinates);
        assert_eq_within_tols(&msj, &-1.0)
    }
    #[test]
    fn valence_3_and_4_noised() -> Result<(), TestError> {
        // https://autotwin.github.io/automesh/cli/metrics_hexahedral.html#unit-tests
        // We test both of the noised elements, valence_03' and valence_04'
        // The source uses ordering [0, 1, 3, 2, 4, 5, 7, 6], so we reorder to standard.

        let mininum_scaled_jacobians_gold = [0.19173666980464177, 0.3743932367172326];

        let nodal_coordinates_set = [
            ElementNodalCoordinates::<N>::from([
                [0.110000e0, 0.120000e0, -0.130000e0],  // 0
                [1.200000e0, -0.200000e0, 0.000000e0],  // 1
                [0.500000e0, 0.866025e0, -0.400000e0],  // 2 (was 3)
                [-0.500000e0, 1.866025e0, -0.200000e0], // 3 (was 2)
                [0.000000e0, 0.000000e0, 1.000000e0],   // 4
                [1.000000e0, 0.000000e0, 1.000000e0],   // 5
                [0.500000e0, 0.866025e0, 1.200000e0],   // 6 (was 7)
                [-0.500000e0, 0.600000e0, 1.400000e0],  // 7 (was 6)
            ]),
            ElementNodalCoordinates::<N>::from([
                [0.100000e0, 0.200000e0, 0.300000e0],   // 0
                [1.200000e0, 0.300000e0, 0.400000e0],   // 1
                [1.030000e0, 1.102000e0, -0.250000e0],  // 2 (was 3)
                [-0.200000e0, 1.200000e0, -0.100000e0], // 3 (was 2)
                [-0.001000e0, -0.021000e0, 1.002000e0], // 4
                [1.200000e0, -0.100000e0, 1.100000e0],  // 5
                [1.010000e0, 1.020000e0, 1.030000e0],   // 6 (was 7)
                [0.000000e0, 1.000000e0, 1.000000e0],   // 7 (was 6)
            ]),
        ];

        nodal_coordinates_set
            .into_iter()
            .zip(mininum_scaled_jacobians_gold)
            .try_for_each(|(nodal_coordinates, msj_gold)| {
                assert_eq_within_tols(
                    &Hexahedron::minimum_scaled_jacobian(nodal_coordinates),
                    &msj_gold,
                )
            })
    }
}
