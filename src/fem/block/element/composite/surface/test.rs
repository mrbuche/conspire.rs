macro_rules! test_composite_surface_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_surface_elements!($element);
        crate::fem::block::element::composite::test::test_composite_element_inner!($element);
        crate::fem::block::element::composite::surface::test::test_composite_surface_element_inner!(
            $element
        );
    };
}
pub(crate) use test_composite_surface_element;

macro_rules! test_composite_surface_element_inner
{
    ($element: ident) =>
    {
        mod composite_surface_element
        {
            use crate::
            {
                fem::block::element::composite::surface::test::test_composite_surface_element_with_constitutive_model,
                math::{Convert, test::{assert_eq_from_fd, assert_eq_within_tols, TestError}}
            };
            use super::*;
            mod elastic
            {
                use crate::constitutive::solid::elastic::AlmansiHamel;
                use super::*;
                mod almansi_hamel
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, AlmansiHamel);
                }
            }
            mod hyperelastic
            {
                use crate::constitutive::solid::hyperelastic::
                    {
                    ArrudaBoyce,
                    Fung,
                    Gent,
                    MooneyRivlin,
                    NeoHookean,
                    SaintVenantKirchoff,
                    Yeoh
                };
                use super::*;
                mod arruda_boyce
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, ArrudaBoyce);
                }
                mod fung
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, Fung);
                }
                mod gent
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, Gent);
                }
                mod mooney_rivlin
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, MooneyRivlin);
                }
                mod neo_hookean
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, NeoHookean);
                }
                mod saint_venant_kirchoff
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, SaintVenantKirchoff);
                }
                mod yeoh
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, Yeoh);
                }
            }
            mod elastic_hyperviscous
            {
                use crate::constitutive::solid::elastic_hyperviscous::AlmansiHamel;
                use super::*;
                mod almansi_hamel
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, AlmansiHamel);
                }
            }
            mod hyperviscoelastic
            {
                use crate::constitutive::solid::hyperviscoelastic::SaintVenantKirchoff;
                use super::*;
                mod saint_venant_kirchoff
                {
                    use super::*;
                    test_composite_surface_element_with_constitutive_model!($element, SaintVenantKirchoff);
                }
            }
        }
    }
}
pub(crate) use test_composite_surface_element_inner;

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
                        &(get_rotation_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &get_coordinates()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_bases(
                        &get_reference_coordinates().into()
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
                        &(get_rotation_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &get_coordinates()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_dual_bases(
                        &get_reference_coordinates().into()
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
                        &(get_rotation_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &get_coordinates()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normals(
                        &get_reference_coordinates().into()
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
                        &(get_rotation_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &get_coordinates()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_gradients(
                        &get_reference_coordinates().into()
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
                                get_coordinates()
                            }
                            else
                            {
                                get_reference_coordinates().into()
                            };
                            nodal_coordinates[a][m] += 0.5 * EPSILON;
                            finite_difference = $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            nodal_coordinates[a][m] -= EPSILON;
                            finite_difference -= $element::<$constitutive_model>::calculate_normals(
                                &nodal_coordinates
                            )[p][i];
                            finite_difference/EPSILON
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
                        &(get_rotation_current_configuration() * get_coordinates()),
                        &(get_rotation_current_configuration() * get_velocities() + get_rotation_rate_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into(),
                        &NodalVelocities::zero()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &get_coordinates(),
                        &get_velocities()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_rates(
                        &get_reference_coordinates().into(),
                        &NodalVelocities::zero()
                    )
                }
            }
        }
        fn get_normal_rates_from_finite_difference(is_deformed: bool) -> NormalRates<P>
        {
            let mut finite_difference = 0.0;
            (0..P).map(|p|
                (0..3).map(|i|
                    get_velocities().iter().enumerate()
                    .map(|(a, velocity_a)|
                        velocity_a.iter().enumerate()
                        .map(|(k, velocity_a_k)|{
                            let mut nodal_coordinates =
                            if is_deformed
                            {
                                get_coordinates()
                            }
                            else
                            {
                                get_reference_coordinates().into()
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
                        &(get_rotation_current_configuration() * get_coordinates())
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &(get_rotation_reference_configuration() * get_reference_coordinates()).into()
                    )
                }
            }
            else
            {
                if is_deformed
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &get_coordinates()
                    )
                }
                else
                {
                    $element::<$constitutive_model>::calculate_normal_tangents(
                        &get_reference_coordinates().into()
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
                                        get_coordinates()
                                    }
                                    else
                                    {
                                        get_reference_coordinates().into()
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
                    &(get_rotation_reference_configuration() * get_reference_coordinates())
                )
            }
            else
            {
                $element::<$constitutive_model>::calculate_reference_normals(
                    &get_reference_coordinates()
                )
            }
        }
    }
}
pub(crate) use setup_for_test_composite_surface_element_with_constitutive_model;

macro_rules! test_composite_surface_element_with_constitutive_model
{
    ($element: ident, $constitutive_model: ident) =>
    {
        setup_for_test_composite_surface_element_with_constitutive_model!($element, $constitutive_model);
        mod bases
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_bases(true, false).iter()
                    .zip(get_bases(true, true).iter())
                    .try_for_each(|(basis, res_basis)|
                        basis.iter()
                        .zip(res_basis.iter())
                        .try_for_each(|(basis_m, res_basis_m)|
                            assert_eq_within_tols(
                                basis_m,
                                &(get_rotation_current_configuration().transpose() * res_basis_m)
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_bases(false, false).iter()
                    .zip(get_bases(false, true).iter())
                    .try_for_each(|(basis, res_basis)|
                        basis.iter()
                        .zip(res_basis.iter())
                        .try_for_each(|(basis_m, res_basis_m)|
                            assert_eq_within_tols(
                                &basis_m.convert(),
                                &(get_rotation_reference_configuration().transpose() * res_basis_m.convert())
                            )
                        )
                    )
                }
            }
        }
        mod dual_bases
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn basis() -> Result<(), TestError>
                {
                    let mut surface_identity = DeformationGradient::identity();
                    surface_identity[2][2] = 0.0;
                    get_bases(true, false).iter()
                    .zip(get_dual_bases(true, false).iter())
                    .try_for_each(|(basis, dual_basis)|
                        assert_eq_within_tols(
                            &basis.iter()
                            .map(|basis|
                                dual_basis.iter()
                                .map(|dual_basis|
                                    basis * dual_basis
                                ).collect()
                            ).collect(),
                            &surface_identity
                        )
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_dual_bases(true, false).iter()
                    .zip(get_dual_bases(true, true).iter())
                    .try_for_each(|(dual_basis, res_dual_basis)|
                        dual_basis.iter()
                        .zip(res_dual_basis.iter())
                        .try_for_each(|(dual_basis_m, res_dual_basis_m)|
                            assert_eq_within_tols(
                                dual_basis_m,
                                &(get_rotation_current_configuration().transpose() * res_dual_basis_m)
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn basis() -> Result<(), TestError>
                {
                    let mut surface_identity = DeformationGradient::identity();
                    surface_identity[2][2] = 0.0;
                    get_bases(false, false).iter()
                    .zip(get_dual_bases(false, false).iter())
                    .try_for_each(|(basis, dual_basis)|
                        assert_eq_within_tols(
                            &basis.iter()
                            .map(|basis|
                                dual_basis.iter()
                                .map(|dual_basis|
                                    basis * dual_basis
                                ).collect()
                            ).collect(),
                            &surface_identity
                        )
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_dual_bases(false, false).iter()
                    .zip(get_dual_bases(false, true).iter())
                    .try_for_each(|(dual_basis, res_dual_basis)|
                        dual_basis.iter()
                        .zip(res_dual_basis.iter())
                        .try_for_each(|(dual_basis_m, res_dual_basis_m)|
                            assert_eq_within_tols(
                                &dual_basis_m.convert(),
                                &(get_rotation_reference_configuration().transpose() * res_dual_basis_m.convert())
                            )
                        )
                    )
                }
            }
        }
        mod normals
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &get_normal_gradients(true, false),
                        &get_normal_gradients_from_finite_difference(true)
                    )
                }
                #[test]
                fn normal() -> Result<(), TestError>
                {
                    get_bases(true, false).iter()
                    .zip(get_normals(true, false).iter())
                    .try_for_each(|(basis, normal)|{
                        assert_eq_within_tols(
                            &(&basis[0] * normal), &0.0
                        )?;
                        assert_eq_within_tols(
                            &(&basis[1] * normal), &0.0
                        )
                    })
                }
                #[test]
                fn normalized() -> Result<(), TestError>
                {
                    get_normals(true, false).iter()
                    .try_for_each(|normal|
                        assert_eq_within_tols(
                            &normal.norm(), &1.0
                        )
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normals(true, false).iter()
                    .zip(get_normals(true, true).iter())
                    .try_for_each(|(normal, res_normal)|
                        assert_eq_within_tols(
                            normal,
                            &(get_rotation_current_configuration().transpose() * res_normal)
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &get_normal_gradients(false, false),
                        &get_normal_gradients_from_finite_difference(false)
                    )
                }
                #[test]
                fn normal() -> Result<(), TestError>
                {
                    get_bases(false, false).iter()
                    .zip(get_normals(false, false).iter())
                    .try_for_each(|(basis, normal)|{
                        assert_eq_within_tols(
                            &(&basis[0] * normal), &0.0
                        )?;
                        assert_eq_within_tols(
                            &(&basis[1] * normal), &0.0
                        )
                    })
                }
                #[test]
                fn normalized() -> Result<(), TestError>
                {
                    get_normals(false, false).iter()
                    .try_for_each(|normal|
                        assert_eq_within_tols(
                            &normal.norm(), &1.0
                        )
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normals(false, false).iter()
                    .zip(get_normals(false, true).iter())
                    .try_for_each(|(normal, res_normal)|
                        assert_eq_within_tols(
                            &normal.convert(),
                            &(get_rotation_reference_configuration().transpose() * res_normal.convert())
                        )
                    )
                }
            }
        }
        mod normal_gradients
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &get_normal_tangents(true, false),
                        &get_normal_tangents_from_finite_difference(true)
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normal_gradients(true, false).iter()
                    .zip(get_normal_gradients(true, true).iter())
                    .try_for_each(|(normal_gradient, res_normal_gradient)|
                        normal_gradient.iter()
                        .zip(res_normal_gradient.iter())
                        .try_for_each(|(normal_gradient_a, res_normal_gradient_a)|
                            assert_eq_within_tols(
                                normal_gradient_a,
                                &(
                                    get_rotation_current_configuration().transpose() *
                                    res_normal_gradient_a *
                                    get_rotation_current_configuration()
                                )
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &get_normal_tangents(false, false),
                        &get_normal_tangents_from_finite_difference(false)
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normal_gradients(false, false).iter()
                    .zip(get_normal_gradients(false, true).iter())
                    .try_for_each(|(normal_gradient, res_normal_gradient)|
                        normal_gradient.iter()
                        .zip(res_normal_gradient.iter())
                        .try_for_each(|(normal_gradient_a, res_normal_gradient_a)|
                            assert_eq_within_tols(
                                &normal_gradient_a.convert(),
                                &(
                                    get_rotation_reference_configuration().transpose() *
                                    res_normal_gradient_a.convert() *
                                    get_rotation_reference_configuration()
                                )
                            )
                        )
                    )
                }
            }
        }
        mod normal_rates
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn finite_difference() -> Result<(), TestError>
                {
                    assert_eq_from_fd(
                        &get_normal_rates(true, false),
                        &get_normal_rates_from_finite_difference(true)
                    )
                }
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normal_rates(true, false).iter()
                    .zip(get_normal_rates(true, true).iter()
                    .zip(get_normals(true, true).iter()))
                    .try_for_each(|(normal_rate, (res_normal_rate, res_normal))|
                        assert_eq_within_tols(
                            normal_rate,
                            &(
                                get_rotation_current_configuration().transpose() *
                                res_normal_rate +
                                get_rotation_rate_current_configuration().transpose() *
                                res_normal
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    get_normal_rates(false, false).iter()
                    .zip(get_normal_rates(false, true).iter())
                    .try_for_each(|(normal_rate, res_normal_rate)|
                        assert_eq_within_tols(
                            &normal_rate.convert(),
                            &(
                                get_rotation_reference_configuration().transpose() *
                                res_normal_rate.convert()
                            )
                        )
                    )
                }
                #[test]
                fn zero() -> Result<(), TestError>
                {
                    assert_eq_within_tols(
                        &get_normal_rates(false, false),
                        &NormalRates::zero()
                    )
                }
            }
        }
        mod normal_tangents
        {
            use super::*;
            mod deformed
            {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    let rotation_transpose = get_rotation_current_configuration().transpose();
                    get_normal_tangents(true, false).iter()
                    .zip(get_normal_tangents(true, true).iter())
                    .try_for_each(|(normal_tangents, res_normal_tangents)|
                        normal_tangents.iter()
                        .zip(res_normal_tangents.iter())
                        .try_for_each(|(normal_tangent_a, res_normal_tangent_a)|
                            normal_tangent_a.iter()
                            .zip(res_normal_tangent_a.iter())
                            .try_for_each(|(normal_tangent_ab, res_normal_tangent_ab)|
                                assert_eq_within_tols(
                                    normal_tangent_ab,
                                    &rotation_transpose.iter()
                                    .map(|rotation_transpose_m|
                                        rotation_transpose.iter()
                                        .map(|rotation_transpose_n|
                                            rotation_transpose.iter()
                                            .map(|rotation_transpose_k|
                                                rotation_transpose_m.iter()
                                                .zip(res_normal_tangent_ab.iter())
                                                .map(|(rotation_transpose_mo, res_normal_tangent_ab_o)|
                                                    rotation_transpose_n.iter()
                                                    .zip(res_normal_tangent_ab_o.iter())
                                                    .map(|(rotation_transpose_np, res_normal_tangent_ab_op)|
                                                        rotation_transpose_k.iter()
                                                        .zip(res_normal_tangent_ab_op.iter())
                                                        .map(|(rotation_transpose_kq, res_normal_tangent_ab_opq)|
                                                            res_normal_tangent_ab_opq * rotation_transpose_mo *
                                                            rotation_transpose_np * rotation_transpose_kq
                                                        ).sum::<Scalar>()
                                                    ).sum::<Scalar>()
                                                ).sum()
                                            ).collect()
                                        ).collect()
                                    ).collect()
                                )
                            )
                        )
                    )
                }
            }
            mod undeformed
            {
                use super::*;
                #[test]
                fn objectivity() -> Result<(), TestError>
                {
                    let rotation_transpose = get_rotation_reference_configuration().transpose();
                    get_normal_tangents(false, false).iter()
                    .zip(get_normal_tangents(false, true).iter())
                    .try_for_each(|(normal_tangents, res_normal_tangents)|
                        normal_tangents.iter()
                        .zip(res_normal_tangents.iter())
                        .try_for_each(|(normal_tangent_a, res_normal_tangent_a)|
                            normal_tangent_a.iter()
                            .zip(res_normal_tangent_a.iter())
                            .try_for_each(|(normal_tangent_ab, res_normal_tangent_ab)|
                                assert_eq_within_tols(
                                    normal_tangent_ab,
                                    &rotation_transpose.iter()
                                    .map(|rotation_transpose_m|
                                        rotation_transpose.iter()
                                        .map(|rotation_transpose_n|
                                            rotation_transpose.iter()
                                            .map(|rotation_transpose_k|
                                                rotation_transpose_m.iter()
                                                .zip(res_normal_tangent_ab.iter())
                                                .map(|(rotation_transpose_mo, res_normal_tangent_ab_o)|
                                                    rotation_transpose_n.iter()
                                                    .zip(res_normal_tangent_ab_o.iter())
                                                    .map(|(rotation_transpose_np, res_normal_tangent_ab_op)|
                                                        rotation_transpose_k.iter()
                                                        .zip(res_normal_tangent_ab_op.iter())
                                                        .map(|(rotation_transpose_kq, res_normal_tangent_ab_opq)|
                                                            res_normal_tangent_ab_opq * rotation_transpose_mo *
                                                            rotation_transpose_np * rotation_transpose_kq
                                                        ).sum::<Scalar>()
                                                    ).sum::<Scalar>()
                                                ).sum()
                                            ).collect()
                                        ).collect()
                                    ).collect()
                                )
                            )
                        )
                    )
                }
            }
        }
        mod reference_normals
        {
            use super::*;
            #[test]
            fn normals() -> Result<(), TestError>
            {
                get_dual_bases(false, false).iter()
                .zip(get_reference_normals(false).iter())
                .try_for_each(|(dual_basis, reference_normal)|{
                    assert_eq_within_tols(
                        &(&dual_basis[0].convert() * reference_normal), &0.0
                    )?;
                    assert_eq_within_tols(
                        &(&dual_basis[1].convert() * reference_normal), &0.0
                    )
                })
            }
            #[test]
            fn objectivity() -> Result<(), TestError>
            {
                get_reference_normals(false).iter()
                .zip(get_reference_normals(true).iter())
                .try_for_each(|(normal, res_normal)|
                    assert_eq_within_tols(
                        normal,
                        &(get_rotation_reference_configuration().transpose() * res_normal)
                    )
                )
            }
        }
    }
}
pub(crate) use test_composite_surface_element_with_constitutive_model;
