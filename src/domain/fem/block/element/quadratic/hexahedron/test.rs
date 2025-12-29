use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
        FiniteElement, GradientVectors,
        quadratic::hexahedron::{G, Hexahedron, M, N},
        solid::SolidFiniteElement,
        test::test_finite_element,
    },
    math::{ScalarList, Tensor},
    mechanics::{DeformationGradient, DeformationGradientList, DeformationGradientRateList},
};

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
    ]
    .into()
}

test_finite_element!(Hexahedron);
