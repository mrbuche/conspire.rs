use crate::{
    constitutive::{
        Constitutive,
        solid::{
            elastic_viscoplastic::{AppliedLoad, ZerothOrderRoot},
            hyperelastic_viscoplastic::Hencky,
        },
    },
    math::{Tensor, integrate::BogackiShampine, optimize::GradientDescent, test::TestError},
};

#[test]
fn foo() -> Result<(), TestError> {
    let (t, f, f_p) = Hencky::new(&[13.0, 3.0, 3.0, 0.25, 1e-1]).root(
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
    for (t_i, (f_i, f_p_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        println!("[{}, {}, {}],", t_i, f_i[0][0], f_p_i[0][0])
    }
    Ok(())
}
