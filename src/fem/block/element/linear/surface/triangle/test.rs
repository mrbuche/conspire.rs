use super::*;
use crate::fem::block::{
    element::linear::surface::test::{
        setup_for_test_finite_element_block_with_elastic_constitutive_model,
        setup_for_test_finite_element_with_elastic_constitutive_model,
        setup_for_test_linear_element_with_constitutive_model,
        setup_for_test_linear_surface_element_with_constitutive_model, test_linear_surface_element,
    },
    test::test_finite_element_block,
};

const D: usize = 16;
const E: usize = 18;

fn get_connectivity() -> Connectivity<E, N> {
    [
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

fn get_coordinates_block() -> NodalCoordinatesBlock {
    NodalCoordinatesBlock::new(&[
        [0.48219277, 0.54126292, 0.03953903],
        [-0.53252101, 0.48863541, 0.02114387],
        [0.11774076, 0.46116171, 0.00839197],
        [-0.12553529, 0.50837336, -0.04498555],
        [-0.51600101, -0.53529173, 0.04709705],
        [-0.48804541, 0.20891774, -0.03607452],
        [-0.51575108, -0.21622038, 0.02508161],
        [0.54161136, -0.53781347, -0.00767790],
        [-0.14439776, -0.53852258, 0.00279141],
        [0.17484086, -0.47179357, 0.04480341],
        [0.46440917, -0.20547688, 0.00791629],
        [0.47547121, 0.16707199, -0.00285113],
        [0.14851037, 0.13199364, 0.04793018],
        [-0.20109496, 0.18518477, 0.04358951],
        [0.12802234, -0.15167375, -0.04216153],
        [-0.21027768, -0.14090996, -0.02122073],
    ])
}

fn get_reference_coordinates() -> ReferenceNodalCoordinates<N> {
    ReferenceNodalCoordinates::new([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
}

fn get_reference_coordinates_block() -> ReferenceNodalCoordinatesBlock {
    ReferenceNodalCoordinatesBlock::new(&[
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [1.0 / 6.0, 0.5, 0.0],
        [-1.0 / 6.0, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [-0.5, 1.0 / 6.0, 0.0],
        [-0.5, -1.0 / 6.0, 0.0],
        [0.5, -0.5, 0.0],
        [-1.0 / 6.0, -0.5, 0.0],
        [1.0 / 6.0, -0.5, 0.0],
        [0.5, -1.0 / 6.0, 0.0],
        [0.5, 1.0 / 6.0, 0.0],
        [1.0 / 6.0, 1.0 / 6.0, 0.0],
        [-1.0 / 6.0, 1.0 / 6.0, 0.0],
        [1.0 / 6.0, -1.0 / 6.0, 0.0],
        [-1.0 / 6.0, -1.0 / 6.0, 0.0],
    ])
}

fn get_velocities_block() -> NodalVelocitiesBlock {
    NodalVelocitiesBlock::new(&[
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

const TEST_SOLVE: bool = false;

fn get_dirichlet_places<'a>() -> [&'a [usize]; 8] {
    panic!()
}

fn get_dirichlet_values(_x: Scalar) -> [Scalar; 8] {
    panic!()
}

test_linear_surface_element!(Triangle);
test_finite_element_block!(Triangle);
