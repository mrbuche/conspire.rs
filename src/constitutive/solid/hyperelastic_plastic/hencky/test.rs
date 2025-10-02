use crate::{
    constitutive::{
        Constitutive,
        solid::{
            elastic_plastic::{AppliedLoad, ElasticPlastic, ZerothOrderRoot},
            hyperelastic_plastic::Hencky,
        },
    },
    math::{Tensor, integrate::BogackiShampine, optimize::GradientDescent, test::TestError},
};

#[test]
fn foo() -> Result<(), TestError> {
    let (t, F, F_p, F_p_dot) = Hencky::new(&[13.0, 3.0]).root(
        AppliedLoad::UniaxialStress(|t| 1.0 + t, &[0.0, 1.0]),
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
    for (t_i, (F_i, F_p_i)) in t.iter().zip(F.iter().zip(F_p.iter())) {
        println!("[{}, {}, {}],", t_i, F_i[0][0], F_p_i[0][0])
    }
    Ok(())
}
