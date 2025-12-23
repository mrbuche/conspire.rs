use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity, FiniteElementBlock,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                GradientVectors,
                linear::wedge::{G, N, Wedge},
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

const D: usize = 21;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [6, 0, 1, 13, 7, 8],
        [6, 5, 0, 13, 12, 7],
        [6, 4, 5, 13, 11, 12],
        [6, 3, 4, 13, 10, 11],
        [6, 2, 3, 13, 9, 10],
        [6, 1, 2, 13, 8, 9],
        [13, 7, 8, 20, 14, 15],
        [13, 12, 7, 20, 19, 14],
        [13, 11, 12, 20, 18, 19],
        [13, 10, 11, 20, 17, 18],
        [13, 9, 10, 20, 16, 17],
        [13, 8, 9, 20, 15, 16],
    ]
}

fn get_coordinates_block() -> NodalCoordinates {
    NodalCoordinates::from([
        [0.06590453, -0.04345507, -1.09785273],
        [-0.02788411, 0.85337004, -0.40775495],
        [0.0122973, 0.88003622, 0.46691946],
        [-0.05751555, -0.00111954, 1.08899018],
        [-0.00777789, -0.90354987, 0.43495861],
        [0.05456049, -0.8077448, -0.55412312],
        [0.05184337, 0.00982936, -0.0187879],
        [0.50434271, -0.09937354, -0.94346274],
        [0.53389247, 0.91259045, -0.52698316],
        [0.40021882, 0.77004218, 0.45037048],
        [0.57038721, -0.00340469, 0.94911495],
        [0.58080652, -0.88357515, 0.44335699],
        [0.5881598, -0.77530513, -0.41622209],
        [0.55889158, 0.05972324, 0.04750797],
        [1.08978362, 0.05291517, -1.03967284],
        [1.00811587, 0.85461345, -0.48484811],
        [1.06576698, 0.95266938, 0.48233746],
        [1.09357093, 0.06907912, 1.09729689],
        [0.90870936, -0.90239317, 0.50676828],
        [0.96597776, -0.76675874, -0.52256089],
        [0.95514861, -0.04083354, -0.08194873],
    ])
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [0.0, 0.0, -1.0],
        [0.0, 3.0_f64.sqrt() / 2.0, -0.5],
        [0.0, 3.0_f64.sqrt() / 2.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, -3.0_f64.sqrt() / 2.0, 0.5],
        [0.0, -3.0_f64.sqrt() / 2.0, -0.5],
        [0.0, 0.0, 0.0],
        [0.5, 0.0, -1.0],
        [0.5, 3.0_f64.sqrt() / 2.0, -0.5],
        [0.5, 3.0_f64.sqrt() / 2.0, 0.5],
        [0.5, 0.0, 1.0],
        [0.5, -3.0_f64.sqrt() / 2.0, 0.5],
        [0.5, -3.0_f64.sqrt() / 2.0, -0.5],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, -1.0],
        [1.0, 3.0_f64.sqrt() / 2.0, -0.5],
        [1.0, 3.0_f64.sqrt() / 2.0, 0.5],
        [1.0, 0.0, 1.0],
        [1.0, -3.0_f64.sqrt() / 2.0, 0.5],
        [1.0, -3.0_f64.sqrt() / 2.0, -0.5],
        [1.0, 0.0, 0.0],
    ])
}

fn get_velocities_block() -> NodalVelocities {
    NodalVelocities::from([
        [-0.00054622, -0.03486379, -0.05312739],
        [-0.0956844, 0.02268205, 0.02409521],
        [0.00449421, 0.02804936, 0.03161703],
        [0.07856535, -0.04662288, -0.0219445],
        [-0.01624672, -0.09556607, 0.00502247],
        [-0.09653708, 0.00231579, -0.07598905],
        [-0.05763119, 0.06726805, -0.0260885],
        [-0.08797888, -0.01771462, 0.04252385],
        [-0.06686631, 0.01599825, 0.08598572],
        [-0.09507766, -0.0743154, -0.00542022],
        [0.01934464, -0.02825188, -0.05982952],
        [-0.08569975, -0.00658211, -0.0335471],
        [0.09176542, -0.03097118, -0.00012169],
        [0.08328906, -0.04431865, 0.06187679],
        [-0.05224908, -0.01607917, -0.00930678],
        [-0.08937773, 0.05135364, 0.05132337],
        [-0.06203837, -0.01876603, 0.07525804],
        [-0.04335809, -0.08143403, 0.08351681],
        [-0.0533697, 0.06013993, -0.04795003],
        [-0.02643142, -0.09931196, 0.08992203],
        [-0.07013487, 0.00545061, -0.02366497],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    let strain = 0.55;
    let mut a = crate::math::Matrix::zero(17, 3 * D);
    a[0][0] = 1.0;
    a[1][3] = 1.0;
    a[2][6] = 1.0;
    a[3][9] = 1.0;
    a[4][12] = 1.0;
    a[5][15] = 1.0;
    a[6][18] = 1.0;
    a[7][42] = 1.0;
    a[8][45] = 1.0;
    a[9][48] = 1.0;
    a[10][51] = 1.0;
    a[11][54] = 1.0;
    a[12][57] = 1.0;
    a[13][60] = 1.0;
    a[14][19] = 1.0;
    a[15][20] = 1.0;
    a[16][1] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 0.0;
    b[1] = 0.0;
    b[2] = 0.0;
    b[3] = 0.0;
    b[4] = 0.0;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 1.0 + strain;
    b[8] = 1.0 + strain;
    b[9] = 1.0 + strain;
    b[10] = 1.0 + strain;
    b[11] = 1.0 + strain;
    b[12] = 1.0 + strain;
    b[13] = 1.0 + strain;
    b[14] = 0.0;
    b[15] = 0.0;
    b[16] = 0.0;
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
    let mut a = crate::math::Matrix::zero(17, 3 * D);
    a[0][0] = 1.0;
    a[1][3] = 1.0;
    a[2][6] = 1.0;
    a[3][9] = 1.0;
    a[4][12] = 1.0;
    a[5][15] = 1.0;
    a[6][18] = 1.0;
    a[7][42] = 1.0;
    a[8][45] = 1.0;
    a[9][48] = 1.0;
    a[10][51] = 1.0;
    a[11][54] = 1.0;
    a[12][57] = 1.0;
    a[13][60] = 1.0;
    a[14][19] = 1.0;
    a[15][20] = 1.0;
    a[16][1] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 0.0;
    b[1] = 0.0;
    b[2] = 0.0;
    b[3] = 0.0;
    b[4] = 0.0;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = velocity;
    b[8] = velocity;
    b[9] = velocity;
    b[10] = velocity;
    b[11] = velocity;
    b[12] = velocity;
    b[13] = velocity;
    b[14] = 0.0;
    b[15] = 0.0;
    b[16] = 0.0;
    (a, b)
}

test_finite_element!(Wedge);
test_finite_element_block!(Wedge);
