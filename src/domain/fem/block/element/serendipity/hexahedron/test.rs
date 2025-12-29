use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity, FiniteElementBlock,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                FiniteElement, GradientVectors,
                serendipity::hexahedron::{G, Hexahedron, M, N},
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

const D: usize = 125;

fn get_connectivity() -> Connectivity<N> {
    todo!()
}

fn get_coordinates_block() -> NodalCoordinates {
    todo!()
}

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
    ]
    .into()
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates {
    todo!()
}

fn get_velocities_block() -> NodalVelocities {
    todo!()
}

fn equality_constraint() -> (
    crate::constitutive::solid::elastic::AppliedLoad,
    crate::math::Matrix,
    crate::math::Vector,
) {
    todo!()
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
    todo!()
}

test_finite_element!(Hexahedron);
test_finite_element_block!(Hexahedron);
