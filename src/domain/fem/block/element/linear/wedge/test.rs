use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
        GradientVectors,
        linear::wedge::{G, N, Wedge},
        solid::SolidFiniteElement,
        test::test_finite_element,
    },
    math::{ScalarList, Tensor},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

const D: usize = 14;

fn get_connectivity() -> Connectivity<N> {
    // vec![
    //     [4, 5, 1, 0, 8],
    //     [5, 6, 2, 1, 8],
    //     [6, 7, 3, 2, 8],
    //     [0, 3, 7, 4, 8],
    //     [0, 1, 2, 3, 8],
    //     [5, 4, 7, 6, 8],
    // ]
    todo!()
}

fn get_coordinates_block() -> NodalCoordinates {
    // NodalCoordinates::from([
    //     [0.04175951, 0.00963520, -0.08547185],
    //     [1.08264022, 0.06657146, -0.06028449],
    //     [1.03545020, 0.95664729, 0.02444034],
    //     [0.03195872, 0.91151568, 0.01357932],
    //     [0.05957727, 0.09722483, 0.95352398],
    //     [1.09602809, 0.05991935, 0.92856463],
    //     [1.00712265, 0.99487330, 0.97093928],
    //     [0.03305756, 1.06846662, 1.02871468],
    //     [0.55951995, 0.55421498, 0.56169451],
    // ])
    todo!()
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
        [0.0, -1.0, 0.0],
        [3.0.sqrt() / 2.0, -1.0 / 2.0, 0.0],
        [3.0.sqrt() / 2.0, 1.0 / 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [-3.0.sqrt() / 2.0, 1.0 / 2.0, 0.0],
        [-3.0.sqrt() / 2.0, -1.0 / 2.0, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, -1.0, 1.0],
        [3.0.sqrt() / 2.0, -1.0 / 2.0, 1.0],
        [3.0.sqrt() / 2.0, 1.0 / 2.0, 1.0],
        [0.0, 1.0, 1.0],
        [-3.0.sqrt() / 2.0, 1.0 / 2.0, 1.0],
        [-3.0.sqrt() / 2.0, -1.0 / 2.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
}

test_finite_element!(Wedge);
