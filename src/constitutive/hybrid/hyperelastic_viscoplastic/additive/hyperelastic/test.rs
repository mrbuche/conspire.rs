use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::{Additive, Multiplicative},
        solid::{
            elastic_viscoplastic::{AppliedLoad, ElasticPlasticOrViscoplastic},
            hyperelastic::{ArrudaBoyce, Hencky},
            hyperelastic_viscoplastic::SecondOrderMinimize,
        },
    },
    math::{Scalar, Tensor, TestError, integrate::BogackiShampine, optimize::NewtonRaphson},
};

#[test]
fn temporary_a() -> Result<(), TestError> {
    // const MU_I: Scalar = 3.85 / 2.0;
    const MU_I: Scalar = 3.85 * 2.0;
    let model = Additive::from((
        Multiplicative::from((
            Hencky {
                bulk_modulus: 460.0,
                shear_modulus: MU_I,
            },
            ViscoplasticFlow {
                yield_stress: 0.077 * MU_I / (1.0 - 0.49),
                hardening_slope: 0.0,
                rate_sensitivity: 1.0,
                // reference_flow_rate: 0.028 / 2.0,
                reference_flow_rate: 0.028 * 2.0,
            },
        )),
        ArrudaBoyce {
            bulk_modulus: 0.0,
            shear_modulus: 5.25 / 3.0,
            number_of_links: 8.0 * 1.45_f64.sqrt(),
        },
    ));
    const RATE: Scalar = 0.1;
    let (t, f, f_p) = model.minimize(
        AppliedLoad::UniaxialStress(
            |t| {
                if RATE * t < 1.0 {
                    1.0 + RATE * t
                } else {
                    3.0 - RATE * t
                }
            },
            &[0.0, 1.8 / RATE],
        ),
        BogackiShampine {
            abs_tol: 1e-6,
            rel_tol: 1e-6,
            ..Default::default()
        },
        NewtonRaphson::default(),
    )?;
    for (t_i, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let (f_p_i, _) = s_i.into();
        let c_e = model.cauchy_stress(f_i, f_p_i)?;
        println!("[{}, {}, {}],", t_i, f_i[0][0] - 1.0, c_e[0][0],)
    }
    Ok(())
}
