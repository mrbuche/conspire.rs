use crate::{
    fem::{
        GradientVectors, NodalCoordinates, NodalForcesSolid, NodalReferenceCoordinates,
        NodalStiffnessesSolid, NodalVelocities, Normals,
        block::{
            Connectivity, ElementBlock, SurfaceFiniteElementBlock,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                SolidFiniteElement, SurfaceFiniteElement, SurfaceFiniteElementMethods,
                linear::triangle::{G, N, P, Triangle},
                test::test_surface_finite_element,
            },
            test::test_surface_finite_element_block,
        },
    },
    math::{Scalar, Scalars, Tensor, optimize::EqualityConstraint},
    mechanics::{
        DeformationGradient, DeformationGradientList, DeformationGradientRate,
        DeformationGradientRateList,
    },
};

const D: usize = 16;

fn get_connectivity() -> Connectivity<N> {
    vec![
        [6, 4, 8],
        [8, 15, 6],
        [15, 8, 9],
        [9, 14, 15],
        [14, 9, 7],
        [7, 10, 14],
        [5, 6, 15],
        [15, 13, 5],
        [13, 15, 14],
        [14, 12, 13],
        [12, 14, 10],
        [10, 11, 12],
        [1, 5, 13],
        [13, 3, 1],
        [3, 13, 12],
        [12, 2, 3],
        [2, 12, 11],
        [11, 0, 2],
    ]
}

fn get_coordinates_block() -> NodalCoordinates {
    NodalCoordinates::from([
        [0.48219277, 0.03953903, 0.54126292],
        [-0.53252101, 0.02114387, 0.48863541],
        [0.11774076, 0.00839197, 0.46116171],
        [-0.12553529, -0.04498555, 0.50837336],
        [-0.51600101, 0.04709705, -0.53529173],
        [-0.48804541, -0.03607452, 0.20891774],
        [-0.51575108, 0.02508161, -0.21622038],
        [0.54161136, -0.00767790, -0.53781347],
        [-0.14439776, 0.00279141, -0.53852258],
        [0.17484086, 0.04480341, -0.47179357],
        [0.46440917, 0.00791629, -0.20547688],
        [0.47547121, -0.00285113, 0.16707199],
        [0.14851037, 0.04793018, 0.13199364],
        [-0.20109496, 0.04358951, 0.18518477],
        [0.12802234, -0.04216153, -0.15167375],
        [-0.21027768, -0.02122073, -0.14090996],
    ])
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5],
        [1.0 / 6.0, 0.0, 0.5],
        [-1.0 / 6.0, 0.0, 0.5],
        [-0.5, 0.0, -0.5],
        [-0.5, 0.0, 1.0 / 6.0],
        [-0.5, 0.0, -1.0 / 6.0],
        [0.5, 0.0, -0.5],
        [-1.0 / 6.0, 0.0, -0.5],
        [1.0 / 6.0, 0.0, -0.5],
        [0.5, 0.0, -1.0 / 6.0],
        [0.5, 0.0, 1.0 / 6.0],
        [1.0 / 6.0, 0.0, 1.0 / 6.0],
        [-1.0 / 6.0, 0.0, 1.0 / 6.0],
        [1.0 / 6.0, 0.0, -1.0 / 6.0],
        [-1.0 / 6.0, 0.0, -1.0 / 6.0],
    ])
}

fn get_velocities_block() -> NodalVelocities {
    NodalVelocities::from([
        [-0.08580606, -0.03719631, -0.06520447],
        [0.07911747, 0.05345331, -0.01990356],
        [-0.06609921, -0.05301467, 0.07700232],
        [-0.06820015, 0.05303888, 0.01960472],
        [0.07911240, -0.05549992, 0.04121606],
        [-0.05686445, -0.03887198, -0.02146579],
        [0.04279461, -0.04073355, -0.00185357],
        [-0.01611562, 0.05904459, -0.06780067],
        [-0.08364077, -0.03687140, 0.05561029],
        [-0.04527588, 0.05764165, 0.06779346],
        [0.05632711, -0.01303029, 0.02199999],
        [0.02326465, -0.03388528, -0.00373330],
        [-0.06275693, 0.08830014, 0.01281029],
        [0.08723950, 0.07024736, 0.04183591],
        [-0.00572455, 0.06721516, -0.00456959],
        [-0.03502294, 0.03342112, 0.00639822],
    ])
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    let strain = 0.55;
    let mut a = crate::math::Matrix::zero(25, 48);
    a[0][0] = 1.0;
    a[1][21] = 1.0;
    a[2][30] = 1.0;
    a[3][33] = 1.0;
    a[4][3] = 1.0;
    a[5][12] = 1.0;
    a[6][15] = 1.0;
    a[7][18] = 1.0;
    //
    a[8][14] = 1.0;
    //
    a[9][1] = 1.0;
    a[10][4] = 1.0;
    a[11][7] = 1.0;
    a[12][10] = 1.0;
    a[13][13] = 1.0;
    a[14][16] = 1.0;
    a[15][19] = 1.0;
    a[16][22] = 1.0;
    a[17][25] = 1.0;
    a[18][28] = 1.0;
    a[19][31] = 1.0;
    a[20][34] = 1.0;
    a[21][37] = 1.0;
    a[22][40] = 1.0;
    a[23][43] = 1.0;
    a[24][46] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = 0.5 + strain;
    b[1] = 0.5 + strain;
    b[2] = 0.5 + strain;
    b[3] = 0.5 + strain;
    b[4] = -0.5;
    b[5] = -0.5;
    b[6] = -0.5;
    b[7] = -0.5;
    //
    b[8] = -0.5;
    //
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
    b[21] = 0.0;
    b[22] = 0.0;
    b[23] = 0.0;
    b[24] = 0.0;
    (
        crate::constitutive::solid::elastic::AppliedLoad::BiaxialStress(strain + 1.0, 1.0),
        a,
        b,
    )
}

fn applied_velocity(
    times: &crate::math::Vector,
) -> crate::constitutive::solid::viscoelastic::AppliedLoad<'_> {
    crate::constitutive::solid::viscoelastic::AppliedLoad::BiaxialStress(
        |_| 0.23,
        |_| 0.0,
        times.as_slice(),
    )
}

fn applied_velocities() -> (crate::math::Matrix, crate::math::Vector) {
    let velocity = 0.23;
    let mut a = crate::math::Matrix::zero(25, 48);
    a[0][0] = 1.0;
    a[1][21] = 1.0;
    a[2][30] = 1.0;
    a[3][33] = 1.0;
    a[4][3] = 1.0;
    a[5][12] = 1.0;
    a[6][15] = 1.0;
    a[7][18] = 1.0;
    //
    a[8][14] = 1.0;
    //
    a[9][1] = 1.0;
    a[10][4] = 1.0;
    a[11][7] = 1.0;
    a[12][10] = 1.0;
    a[13][13] = 1.0;
    a[14][16] = 1.0;
    a[15][19] = 1.0;
    a[16][22] = 1.0;
    a[17][25] = 1.0;
    a[18][28] = 1.0;
    a[19][31] = 1.0;
    a[20][34] = 1.0;
    a[21][37] = 1.0;
    a[22][40] = 1.0;
    a[23][43] = 1.0;
    a[24][46] = 1.0;
    let mut b = crate::math::Vector::zero(a.len());
    b[0] = velocity;
    b[1] = velocity;
    b[2] = velocity;
    b[3] = velocity;
    b[4] = 0.0;
    b[5] = 0.0;
    b[6] = 0.0;
    b[7] = 0.0;
    //
    b[8] = 0.0;
    //
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
    b[21] = 0.0;
    b[22] = 0.0;
    b[23] = 0.0;
    b[24] = 0.0;
    (a, b)
}

test_surface_finite_element!(Triangle);
test_surface_finite_element_block!(Triangle);
