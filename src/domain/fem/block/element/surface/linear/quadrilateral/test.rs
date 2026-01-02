use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block, Connectivity,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                FiniteElement, GradientVectors,
                solid::SolidFiniteElement,
                surface::{
                    Normals, SurfaceFiniteElement,
                    linear::{
                        quadrilateral::{G, M, N, Quadrilateral},
                        triangle::test::{
                            D, applied_velocities, applied_velocity, equality_constraint,
                            get_coordinates_block, get_reference_coordinates_block,
                            get_velocities_block,
                        },
                    },
                },
                test::test_surface_finite_element,
            },
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
            test::test_surface_finite_element_block,
        },
    },
    math::{Scalar, ScalarList, Tensor, optimize::EqualityConstraint},
    mechanics::{
        DeformationGradient, DeformationGradientList, DeformationGradientRate,
        DeformationGradientRateList,
    },
};

fn get_connectivity() -> Connectivity<N> {
    vec![
        [5, 13, 3, 1],
        [13, 12, 2, 3],
        [12, 11, 0, 2],
        [6, 15, 13, 5],
        [15, 14, 12, 13],
        [14, 10, 11, 12],
        [4, 8, 15, 6],
        [8, 9, 14, 15],
        [9, 7, 10, 14],
    ]
}

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    ElementNodalReferenceCoordinates::from([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
}

test_surface_finite_element!(Quadrilateral);
test_surface_finite_element_block!(Quadrilateral);
