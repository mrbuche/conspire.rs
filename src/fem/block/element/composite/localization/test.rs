macro_rules! test_composite_localization_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_localization_elements!($element);
        crate::fem::block::element::composite::test::test_composite_element_inner!($element);
        crate::fem::block::element::composite::surface::test::test_composite_surface_element_inner!(
            $element
        );
        fn get_coordinates_unrotated() -> NodalCoordinates<N> {
            let jump = get_jump();
            let mut coordinates = get_deformation_gradient_surface() * get_reference_coordinates();
            coordinates
                .iter_mut()
                .skip(3)
                .take(3)
                .for_each(|coordinate_top_a| *coordinate_top_a += &jump);
            coordinates
                .iter_mut()
                .skip(9)
                .take(3)
                .for_each(|coordinate_top_a| *coordinate_top_a += &jump);
            coordinates
        }
        fn get_velocities_unrotated() -> NodalVelocities<N> {
            let jump_rate = get_jump_rate();
            let mut velocities =
                get_deformation_gradient_rate_surface() * get_reference_coordinates();
            velocities
                .iter_mut()
                .skip(3)
                .take(3)
                .for_each(|velocity_top_a| *velocity_top_a += &jump_rate);
            velocities
                .iter_mut()
                .skip(9)
                .take(3)
                .for_each(|velocity_top_a| *velocity_top_a += &jump_rate);
            velocities
        }
    };
}
pub(crate) use test_composite_localization_element;

macro_rules! setup_for_test_composite_element_with_constitutive_model {
    ($element: ident, $constitutive_model: ident, $constitutive_model_parameters: ident) => {
        fn get_element<'a>() -> $element<$constitutive_model<'a>> {
            $element::new(
                $constitutive_model_parameters,
                get_reference_coordinates(),
                &crate::fem::block::element::linear::surface::test::THICKNESS,
            )
        }
        fn get_element_transformed<'a>() -> $element<$constitutive_model<'a>> {
            $element::new(
                $constitutive_model_parameters,
                get_reference_coordinates_transformed(),
                &crate::fem::block::element::linear::surface::test::THICKNESS,
            )
        }
        #[test]
        fn size() {
            assert_eq!(
                std::mem::size_of::<$element::<$constitutive_model>>(),
                std::mem::size_of::<[$constitutive_model; G]>()
                    + std::mem::size_of::<ProjectedGradientVectors<G, N>>()
                    + std::mem::size_of::<Scalars<G>>()
                    + std::mem::size_of::<ScaledReferenceNormals<G, P>>()
            )
        }
        #[test]
        #[should_panic]
        fn calculate_projected_gradient_vectors() {
            $element::<$constitutive_model>::calculate_projected_gradient_vectors(
                &$element::<$constitutive_model>::calculate_midplane(&get_reference_coordinates())
                    .into(),
            );
        }
    };
}
pub(crate) use setup_for_test_composite_element_with_constitutive_model;

macro_rules! setup_for_test_composite_surface_element_with_constitutive_model
{
    ($element: ident, $constitutive_model: ident) =>
    {
        fn get_bases(is_deformed: bool, is_transformed: bool) -> Bases<1, P>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        )
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates().into()
                        )
                    )
                }
            }
        }
        fn get_dual_bases(is_deformed: bool, is_transformed: bool) -> Bases<1, P>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        )
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates().into()
                        )
                    )
                }
            }
        }
        fn get_normals(is_deformed: bool, is_transformed: bool) -> Normals<P>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_current_configuration() * get_coordinates())
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        )
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates().into()
                        )
                    )
                }
            }
        }
        fn get_normal_gradients(is_deformed: bool, is_transformed: bool) -> NormalGradientss<P, O>
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
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        )
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
                            &get_reference_coordinates().into()
                        )
                    )
                }
            }
        }
        fn get_normal_gradients_from_finite_difference(is_deformed: bool) -> NormalGradientss<P, O>
        {
            let mut finite_difference = 0.0;
            (0..P).map(|p|
                (0..O).map(|a|
                    (0..3).map(|m|
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
                                    &get_reference_coordinates().into()
                                )
                            };
                            nodal_coordinates[a][m] += 0.5 * EPSILON;
                            finite_difference = $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            nodal_coordinates[a][m] -= EPSILON;
                            finite_difference -= $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            finite_difference / EPSILON
                        }).collect()
                    ).collect()
                ).collect()
            ).collect()
        }
        fn get_normal_rates(is_deformed: bool, is_transformed: bool) -> NormalRates<P>
        {
            if is_transformed
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
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
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        ),
                        &$element::<$constitutive_model>::calculate_midplane(
                            &NodalVelocities::zero()
                        )
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_coordinates()
                        ),
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_velocities()
                        )
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &$element::<$constitutive_model>::calculate_midplane(
                            &get_reference_coordinates().into()
                        ),
                        &$element::<$constitutive_model>::calculate_midplane(
                            &NodalVelocities::zero()
                        )
                    )
                }
            }
        }
        fn get_normal_rates_from_finite_difference(is_deformed: bool) -> NormalRates<P>
        {
            let mut finite_difference = 0.0;
            (0..P).map(|p|
                (0..3).map(|i|
                    $element::<$constitutive_model>::calculate_midplane(&get_velocities()).iter().enumerate()
                    .map(|(a, velocity_a)|
                        velocity_a.iter().enumerate()
                        .map(|(k, velocity_a_k)|{
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
                                    &get_reference_coordinates().into()
                                )
                            };
                            nodal_coordinates[a][k] += 0.5 * EPSILON;
                            finite_difference = $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            nodal_coordinates[a][k] -= EPSILON;
                            finite_difference -= $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            finite_difference/EPSILON * velocity_a_k
                        }).sum::<Scalar>()
                    ).sum()
                ).collect()
            ).collect()
        }
        fn get_normal_tangents(is_deformed: bool, is_transformed: bool) -> NormalTangentss<P, O>
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
                            &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                        )
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
                            &get_reference_coordinates().into()
                        )
                    )
                }
            }
        }
        fn get_normal_tangents_from_finite_difference(is_deformed: bool) -> NormalTangentss<P, O>
        {
            let mut finite_difference = 0.0;
            (0..P).map(|p|
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
                                            &get_reference_coordinates().into()
                                        )
                                    };
                                    nodal_coordinates[b][n] += 0.5 * EPSILON;
                                    finite_difference = $element::<$constitutive_model>::calculate_normal_gradients(
                                        &nodal_coordinates
                                    )[p][a][m][i];
                                    nodal_coordinates[b][n] -= EPSILON;
                                    finite_difference -= $element::<$constitutive_model>::calculate_normal_gradients(
                                        &nodal_coordinates
                                    )[p][a][m][i];
                                    finite_difference/EPSILON
                                }).collect()
                            ).collect()
                        ).collect()
                    ).collect()
                ).collect()
            ).collect()
        }
        fn get_reference_normals(is_transformed: bool) -> ReferenceNormals<P>
        {
            if is_transformed
            {
                $element::<$constitutive_model>::calculate_reference_normals(
                    &$element::<$constitutive_model>::calculate_midplane(
                        &(get_rotation_reference_configuration() * get_reference_coordinates())
                    )
                )
            }
            else
            {
                $element::<$constitutive_model>::calculate_reference_normals(
                    &$element::<$constitutive_model>::calculate_midplane(
                        &get_reference_coordinates()
                    )
                )
            }
        }
    }
}
pub(crate) use setup_for_test_composite_surface_element_with_constitutive_model;
