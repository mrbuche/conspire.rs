use crate::{
    constitutive::{
        Constitutive,
        solid::{
            elastic_viscoplastic::{AppliedLoad, ElasticViscoplastic},
            hyperelastic_viscoplastic::Hencky,
        },
    },
    math::{
        Rank2, Tensor, TensorArray,
        integrate::BogackiShampine,
        optimize::{GradientDescent, NewtonRaphson},
        test::{ErrorTensor, TestError, assert_eq_from_fd},
    },
    mechanics::{CauchyTangentStiffness, DeformationGradient, DeformationGradientPlastic},
};

// if constitutive models are never going hold state variables
// i.e. they are always going to be immutable
// (1) can rethink their construction, i.e. can make fields public and make like solvers (also better for debug prints; can use From for make from slices still)
// (2) can rethink how they are used in fem, i.e. can have elements store a reference to a model rather than own it (but still leave room for spatial variance)

#[test]
fn finite_difference() -> Result<(), TestError> {
    let deformation_gradient = DeformationGradient::new([
        [1.31924942, 1.36431217, 0.41764434],
        [0.09959341, 1.38409741, 1.48320137],
        [0.21114106, 1.16675104, 1.98146028],
    ]);
    let deformation_gradient_p = DeformationGradientPlastic::new([
        [0.79610657, 1.36265438, 0.58765375],
        [0.71714877, 1.83110678, 0.69670465],
        [1.82260662, 2.1921719, 3.16928404],
    ]);
    let model = Hencky::new(&[13.0, 3.0, 3.0, 0.0, 0.25, 0.1]);
    let tangent = model.cauchy_tangent_stiffness(&deformation_gradient, &deformation_gradient_p)?;
    let mut fd = CauchyTangentStiffness::zero();
    for k in 0..3 {
        for l in 0..3 {
            let mut deformation_gradient_plus = deformation_gradient.clone();
            deformation_gradient_plus[k][l] += 0.5 * crate::EPSILON;
            let cauchy_stress_plus =
                model.cauchy_stress(&deformation_gradient_plus, &deformation_gradient_p)?;
            let mut deformation_gradient_minus = deformation_gradient.clone();
            deformation_gradient_minus[k][l] -= 0.5 * crate::EPSILON;
            let cauchy_stress_minus =
                model.cauchy_stress(&deformation_gradient_minus, &deformation_gradient_p)?;
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j][k][l] =
                        (cauchy_stress_plus[i][j] - cauchy_stress_minus[i][j]) / crate::EPSILON;
                }
            }
        }
    }
    if tangent.error_fd(&fd, &(5e1 * crate::EPSILON)).is_some() {
        assert_eq_from_fd(&tangent, &fd)
    } else {
        Ok(())
    }
}

#[test]
fn root_0() -> Result<(), TestError> {
    use crate::constitutive::solid::elastic_viscoplastic::ZerothOrderRoot;
    let model = Hencky::new(&[13.0, 3.0, 3.0, 0.0, 0.05, 0.1]);
    let (t, f, f_p) = model.root(
        AppliedLoad::UniaxialStress(|t| 1.0 + t, &[0.0, 8.0]),
        BogackiShampine {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..Default::default()
        },
        GradientDescent {
            dual: true,
            ..Default::default()
        },
    )?;
    for (t_i, (f_i, f_p_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let f_e = f_i * f_p_i.inverse();
        let m_e = f_e.transpose() * model.cauchy_stress(&f_i, &f_p_i)? * f_e.inverse_transpose();
        let m_e_dev_mag = m_e.deviatoric().norm();
        println!(
            "[{}, {}, {}, {}, {}],",
            t_i,
            f_i[0][0],
            f_p_i[0][0],
            f_p_i.determinant(),
            m_e_dev_mag,
        )
    }
    Ok(())
}

#[test]
fn root_1() -> Result<(), TestError> {
    use crate::constitutive::solid::elastic_viscoplastic::FirstOrderRoot;
    let model = Hencky::new(&[13.0, 3.0, 3.0, 0.0, 0.05, 0.1]);
    let (t, f, f_p) = model.root(
        AppliedLoad::UniaxialStress(|t| 1.0 + t, &[0.0, 8.0]),
        BogackiShampine {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..Default::default()
        },
        NewtonRaphson::default(),
    )?;
    for (t_i, (f_i, f_p_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let f_e = f_i * f_p_i.inverse();
        let m_e = f_e.transpose() * model.cauchy_stress(&f_i, &f_p_i)? * f_e.inverse_transpose();
        let m_e_dev_mag = m_e.deviatoric().norm();
        println!(
            "[{}, {}, {}, {}, {}],",
            t_i,
            f_i[0][0],
            f_p_i[0][0],
            f_p_i.determinant(),
            m_e_dev_mag,
        )
    }
    Ok(())
}
