use super::*;
use crate::{
    fem::block::{element::test::test_finite_element, test::test_finite_element_block},
    math::TensorArray,
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
    NodalCoordinatesBlock::new(&[
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
    ReferenceNodalCoordinatesBlock::new(&[
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
    NodalVelocitiesBlock::new(&[
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

const TEST_SOLVE: bool = true;

fn get_dirichlet_places<'a>() -> [&'a [usize]; 10] {
    [
        &[0, 0],
        &[1, 0],
        &[2, 0],
        &[3, 0],
        &[4, 0],
        &[5, 0],
        &[6, 0],
        &[7, 0],
        &[11, 0],
        &[13, 0],
    ]
}

fn get_dirichlet_values(x: Scalar) -> [Scalar; 10] {
    [
        0.5 + x,
        0.5 + x,
        -0.5,
        -0.5,
        0.5 + x,
        0.5 + x,
        -0.5,
        -0.5,
        -0.5,
        0.5 + x,
    ]
}

test_finite_element!(Tetrahedron);
test_finite_element_block!(Tetrahedron);
