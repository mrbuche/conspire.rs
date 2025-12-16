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

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::new([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
}

test_finite_element!(Wedge);
