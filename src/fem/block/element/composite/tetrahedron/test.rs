use super::*;
use crate::fem::block::{
    element::{
        composite::test::{
            setup_for_test_composite_element_with_constitutive_model, test_composite_element,
        },
        test::setup_for_test_finite_element_with_elastic_constitutive_model,
    },
    test::{
        setup_for_test_finite_element_block_with_elastic_constitutive_model,
        test_finite_element_block,
    },
};

const D: usize = 35;
const E: usize = 12;

fn get_connectivity() -> Connectivity<E, N> {
    [
        [26, 1, 3, 0, 27, 20, 28, 29, 13, 10],
        [26, 2, 3, 1, 30, 19, 28, 27, 9, 20],
        [26, 5, 4, 1, 31, 8, 32, 27, 22, 12],
        [26, 7, 4, 6, 33, 16, 32, 34, 18, 25],
        [26, 2, 1, 4, 30, 9, 27, 32, 23, 12],
        [26, 5, 1, 0, 31, 22, 27, 29, 11, 13],
        [26, 7, 2, 4, 33, 17, 30, 32, 16, 23],
        [26, 6, 0, 3, 34, 21, 29, 28, 14, 10],
        [26, 6, 5, 0, 34, 15, 31, 29, 21, 11],
        [26, 6, 4, 5, 34, 25, 32, 31, 15, 8],
        [3, 7, 2, 26, 24, 17, 19, 28, 33, 30],
        [3, 6, 7, 26, 14, 18, 24, 28, 34, 33],
    ]
}

fn get_coordinates_block() -> NodalCoordinatesBlock {
    NodalCoordinatesBlock::new(&[
        [0.50970092, -0.45999746, 0.47715613],
        [0.53320092, 0.50645170, 0.48671275],
        [-0.48918872, 0.49235727, 0.52419583],
        [-0.49988814, -0.49928014, 0.53093352],
        [0.47228840, 0.53108429, -0.53544051],
        [0.54676338, -0.51413459, -0.46964184],
        [-0.47139791, -0.46137005, -0.51494158],
        [-0.45634554, 0.51673544, -0.45894451],
        [0.49343959, -0.04586137, -0.51780713],
        [0.02073785, 0.45583393, 0.53994113],
        [0.04953278, -0.54512259, 0.48544961],
        [0.45473887, -0.48393625, -0.02131588],
        [0.50705760, 0.45245186, 0.03884302],
        [0.54064179, -0.02744916, 0.50086114],
        [-0.46039517, -0.52661921, -0.03376956],
        [-0.04428896, -0.50361475, -0.47149995],
        [0.03195679, 0.52012652, -0.54432589],
        [-0.47816797, 0.45276849, 0.04170033],
        [-0.46628810, -0.00745756, -0.53050943],
        [-0.52963705, -0.02523134, 0.50911825],
        [0.03325220, 0.04511703, 0.47540805],
        [0.04259935, -0.46183526, 0.03719249],
        [0.52885027, -0.01896105, -0.00977622],
        [0.00205826, 0.54671803, -0.03295283],
        [-0.45405149, 0.03799887, -0.02202679],
        [-0.04436629, -0.04125653, -0.49811086],
        [0.01094444, -0.03187500, 0.02330344],
        [0.20207504, 0.26580515, 0.25272273],
        [-0.27097198, -0.21226878, 0.23326076],
        [0.24242429, -0.24775049, 0.28098121],
        [-0.20547580, 0.23878784, 0.20319491],
        [0.24273166, -0.22552909, -0.27635205],
        [0.23046696, 0.20175434, -0.24036472],
        [-0.23621308, 0.22857201, -0.26962741],
        [-0.28625675, -0.27375433, -0.26379361],
    ])
}

fn get_reference_coordinates() -> ReferenceNodalCoordinates<N> {
    ReferenceNodalCoordinates::new([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
}

fn get_reference_coordinates_block() -> ReferenceNodalCoordinatesBlock {
    ReferenceNodalCoordinatesBlock::new(&[
        [0.50, -0.50, 0.50],
        [0.50, 0.50, 0.50],
        [-0.50, 0.50, 0.50],
        [-0.50, -0.50, 0.50],
        [0.50, 0.50, -0.50],
        [0.50, -0.50, -0.50],
        [-0.50, -0.50, -0.50],
        [-0.50, 0.50, -0.50],
        [0.50, 0.00, -0.50],
        [0.00, 0.50, 0.50],
        [0.00, -0.50, 0.50],
        [0.50, -0.50, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [-0.50, -0.50, 0.00],
        [0.00, -0.50, -0.50],
        [0.00, 0.50, -0.50],
        [-0.50, 0.50, 0.00],
        [-0.50, 0.00, -0.50],
        [-0.50, 0.00, 0.50],
        [0.00, 0.00, 0.50],
        [0.00, -0.50, 0.00],
        [0.50, 0.00, 0.00],
        [0.00, 0.50, 0.00],
        [-0.50, 0.00, 0.00],
        [0.00, 0.00, -0.50],
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [-0.25, -0.25, 0.25],
        [0.25, -0.25, 0.25],
        [-0.25, 0.25, 0.25],
        [0.25, -0.25, -0.25],
        [0.25, 0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, -0.25],
    ])
}

fn get_velocities_block() -> NodalVelocitiesBlock {
    NodalVelocitiesBlock::new(&[
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
        [-0.01017102, 0.02293921, 0.01072471],
        [-0.07521029, -0.01340458, -0.03314643],
        [0.04582758, -0.04733806, -0.03788661],
        [-0.09159262, -0.05441244, 0.00583849],
        [0.05529252, -0.06128855, 0.02814717],
        [-0.06432809, -0.05701089, -0.08241133],
        [0.02712482, -0.09724979, 0.02285835],
        [-0.05965081, -0.07478612, -0.04896525],
    ])
}

const TEST_SOLVE: bool = false;

fn get_dirichlet_places<'a>() -> [&'a [usize]; 8] {
    panic!()
}

fn get_dirichlet_values(_x: Scalar) -> [Scalar; 8] {
    panic!()
}

test_composite_element!(Tetrahedron);
test_finite_element_block!(Tetrahedron);
