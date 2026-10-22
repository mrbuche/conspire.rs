use crate::{
    fem::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block,
            element::{
                ElementNodalCoordinates, ElementNodalReferenceCoordinates, ElementNodalVelocities,
                FiniteElementInterpolation, GradientVectors,
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
            test::test_surface_finite_element_block,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{ScalarList, Tensor},
    mechanics::{DeformationGradient, DeformationGradientRate},
};

fn get_connectivity() -> Vec<[usize; N]> {
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
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    .into()
}

test_surface_finite_element!(Quadrilateral);
test_surface_finite_element_block!(Quadrilateral);

mod traction_forces {
    use super::{N, Quadrilateral};
    use crate::{
        fem::block::element::surface::SurfaceFiniteElement,
        math::assert::{Assert, AssertionError},
    };

    #[test]
    fn share_the_traction_of_the_area_among_the_nodes() -> Result<(), AssertionError> {
        let forces = Quadrilateral::traction_forces(
            &[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 3.0, 1.0],
                [0.0, 3.0, 0.0],
            ]
            .into(),
            &[1.0, 2.0, 3.0].into(),
        );
        let share = 45.0_f64.sqrt() / 4.0;
        Assert::default().eq_within_tols(&forces, &[[share, 2.0 * share, 3.0 * share]; N].into())
    }
}
