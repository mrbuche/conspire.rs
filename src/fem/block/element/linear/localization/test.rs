macro_rules! test_linear_localization_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_localization_elements!($element);
        crate::fem::block::element::linear::test::test_linear_element_inner!($element);
        crate::fem::block::element::linear::surface::test::test_linear_surface_element_inner!(
            $element
        );
        fn get_coordinates_unrotated() -> NodalCoordinates<N> {
            let jump = get_jump();
            let mut coordinates = get_deformation_gradient_surface() * get_reference_coordinates();
            coordinates
                .iter_mut()
                .skip(O)
                .for_each(|coordinate_top_a| *coordinate_top_a += &jump);
            coordinates
        }
        fn get_velocities_unrotated() -> NodalVelocities<N> {
            let jump_rate = get_jump_rate();
            let mut velocities =
                get_deformation_gradient_rate_surface() * get_reference_coordinates();
            velocities
                .iter_mut()
                .skip(O)
                .for_each(|velocity_top_a| *velocity_top_a += &jump_rate);
            velocities
        }
    };
}
pub(crate) use test_linear_localization_element;

macro_rules! setup_for_test_linear_surface_element_with_constitutive_model
{
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) =>
    {
        fn get_basis(is_deformed: bool, is_transformed: bool) -> Basis<1>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_basis(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_basis(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_basis(
                        &$element::<$constitutive_model>::calculate_midplane(&get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_basis(
                        &$element::<$constitutive_model>::calculate_midplane(&get_reference_coordinates()).into()
                    )
                }
            }
        }
        fn get_dual_basis(is_deformed: bool, is_transformed: bool) -> Basis<1>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_dual_basis(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_basis(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_dual_basis(
                        &$element::<$constitutive_model>::calculate_midplane(&get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_basis(
                        &$element::<$constitutive_model>::calculate_midplane(&get_reference_coordinates()).into()
                    )
                }
            }
        }
        fn get_normal(is_deformed: bool, is_transformed: bool) -> Normal
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates()
                        ).into()
                    )
                }
            }
        }
        fn get_normal_gradients(is_deformed: bool, is_transformed: bool) -> NormalGradients<O>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates()
                        ).into()
                    )
                }
            }
        }
        fn get_normal_gradients_from_finite_difference(is_deformed: bool) -> NormalGradients<O>
        {
            let mut finite_difference = 0.0;
            (0..O).map(|a|
                (0..3).map(|i|
                    (0..3).map(|j|{
                        let mut nodal_coordinates =
                        if is_deformed
                        {
                            $element::<$constitutive_model>::calculate_midplane(
                                &get_coordinates()
                            )
                        }
                        else
                        {
                            $element::<$constitutive_model>::calculate_midplane(
                                &get_reference_coordinates()
                            ).into()
                        };
                        nodal_coordinates[a][i] += 0.5 * EPSILON;
                        finite_difference = $element::<$constitutive_model>::calculate_normal(
                            &nodal_coordinates
                        )[j];
                        nodal_coordinates[a][i] -= EPSILON;
                        finite_difference -= $element::<$constitutive_model>::calculate_normal(
                            &nodal_coordinates
                        )[j];
                        finite_difference/EPSILON
                    }).collect()
                ).collect()
            ).collect()
        }
        fn get_normal_rate(is_deformed: bool, is_transformed: bool) -> NormalRate
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_rate(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        ),
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_velocities() + get_rotation_rate_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_rate(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into(),
                        &NodalVelocities::zero()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_rate(
                        &$element::<$constitutive_model>::calculate_midplane(&get_coordinates()),
                        &$element::<$constitutive_model>::calculate_midplane(&get_velocities())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_rate(
                        &$element::<$constitutive_model>::calculate_midplane(&get_reference_coordinates()).into(),
                        &NodalVelocities::zero()
                    )
                }
            }
        }
        fn get_normal_rate_from_finite_difference(is_deformed: bool) -> NormalRate
        {
            let mut finite_difference = 0.0;
            (0..3).map(|i|
                $element::<$constitutive_model>::calculate_midplane(&get_velocities()).iter().enumerate()
                .map(|(a, velocity_a)|
                    velocity_a.iter().enumerate()
                    .map(|(k, velocity_a_k)|{
                        let mut nodal_coordinates =
                        if is_deformed
                        {
                            $element::<$constitutive_model>::calculate_midplane(&get_coordinates())
                        }
                        else
                        {
                            $element::<$constitutive_model>::calculate_midplane(&get_reference_coordinates()).into()
                        };
                        nodal_coordinates[a][k] += 0.5 * EPSILON;
                        finite_difference = $element::<$constitutive_model>::calculate_normal(
                            &nodal_coordinates
                        )[i];
                        nodal_coordinates[a][k] -= EPSILON;
                        finite_difference -= $element::<$constitutive_model>::calculate_normal(
                            &nodal_coordinates
                        )[i];
                        finite_difference/EPSILON * velocity_a_k
                    }).sum::<Scalar>()
                ).sum()
            ).collect()
        }
        fn get_normal_tangents(is_deformed: bool, is_transformed: bool) -> NormalTangents<O>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates())
                        ).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates()
                        ).into()
                    )
                }
            }
        }
        fn get_normal_tangents_from_finite_difference(is_deformed: bool) -> NormalTangents<O>
        {
            let mut finite_difference = 0.0;
            (0..O).map(|a|
                (0..O).map(|b|
                    (0..3).map(|m|
                        (0..3).map(|n|
                            (0..3).map(|i|{
                                let mut nodal_coordinates =
                                if is_deformed
                                {
                                    $element::<$constitutive_model>::calculate_midplane(
                                        &get_coordinates()
                                    )
                                }
                                else
                                {
                                    $element::<$constitutive_model>::calculate_midplane(
                                        &get_reference_coordinates()
                                    ).into()
                                };
                                nodal_coordinates[b][n] += 0.5 * EPSILON;
                                finite_difference = $element::<$constitutive_model>::calculate_normal_gradients(
                                    &nodal_coordinates
                                )[a][m][i];
                                nodal_coordinates[b][n] -= EPSILON;
                                finite_difference -= $element::<$constitutive_model>::calculate_normal_gradients(
                                    &nodal_coordinates
                                )[a][m][i];
                                finite_difference/EPSILON
                            }).collect()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect()
        }
        #[test]
        fn size()
        {
            assert_eq!(
                std::mem::size_of::<$element::<$constitutive_model>>(),
                std::mem::size_of::<$constitutive_model>()
                + std::mem::size_of::<GradientVectors<N>>()
                + std::mem::size_of::<ReferenceNormal>()
                + std::mem::size_of::<Scalar>()
            )
        }
        #[test]
        #[should_panic]
        fn calculate_gradient_vectors()
        {
            $element::<$constitutive_model>::calculate_gradient_vectors(
                &$element::<$constitutive_model>::calculate_midplane(
                    &get_reference_coordinates()
                ).into()
            );
        }
    }
}
pub(crate) use setup_for_test_linear_surface_element_with_constitutive_model;
