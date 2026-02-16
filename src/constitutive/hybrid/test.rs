use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::{
            ElasticMultiplicativeViscoplastic, ElasticViscoplasticAdditiveElastic,
            ElasticViscoplasticAdditiveViscoplastic,
        },
        solid::{
            elastic_viscoplastic::{AppliedLoad, ElasticPlasticOrViscoplastic, FirstOrderRoot},
            hyperelastic::{Hencky, NeoHookean},
        },
    },
    math::{Tensor, integrate::DormandPrince, optimize::NewtonRaphson, test::TestError},
    mechanics::Scalar,
};

#[test]
fn demonstration() -> Result<(), TestError> {
    let rate_sensitivity = 1.0;
    let reference_flow_rate = 0.008; // 6.72, need to make ActivatedViscoplasticFlow model with exp(-dG/kT)sinh(dG/kT S/Y) and Y_dot features
    let model = ElasticViscoplasticAdditiveElastic::from((
        ElasticMultiplicativeViscoplastic::from((
            Hencky {
                bulk_modulus: 330.0,
                shear_modulus: 110.0,
            },
            ElasticViscoplasticAdditiveViscoplastic::from((
                ElasticMultiplicativeViscoplastic::from((
                    Hencky {
                        bulk_modulus: 0.0,
                        shear_modulus: 26.5,
                    },
                    ViscoplasticFlow {
                        yield_stress: 4.3,
                        hardening_slope: 230.0,
                        rate_sensitivity,
                        reference_flow_rate,
                    },
                )),
                ViscoplasticFlow {
                    yield_stress: 6.5,
                    hardening_slope: 26.0,
                    rate_sensitivity,
                    reference_flow_rate,
                },
            )),
        )),
        NeoHookean {
            bulk_modulus: 0.0,
            shear_modulus: 3.3,
        },
    ));
    const RATE: Scalar = 0.01;
    let (t, f, f_p) = model.root(
        AppliedLoad::UniaxialStress(
            |t| {
                if RATE * t <= 0.1 {
                    1.0 + RATE * t
                } else if RATE * t <= 0.1 + 0.045 {
                    (1.0 + 0.1) - (RATE * t - 0.1)
                } else if RATE * t <= 0.2 + 2.0 * 0.045 {
                    (1.0 + 0.1) - (0.1 + 0.045 - 0.1) + (RATE * t - (0.1 + 0.045))
                } else if RATE * t <= 0.2 + 2.0 * 0.045 + 0.075 {
                    1.2 - (RATE * t - (0.2 + 2.0 * 0.045))
                } else {
                    1.0
                }
            },
            &[0.0, (0.2 + 2.0 * 0.045 + 0.075) / RATE],
        ),
        DormandPrince::default(),
        NewtonRaphson::default(),
    )?;
    for (t_i, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let (f_p_i, _) = s_i.into();
        let c_e = model.cauchy_stress(f_i, f_p_i)?;
        println!(
            "[{}, {}, {}, {}],",
            t_i,
            f_i[0][0],
            f_i[0][0].ln(),
            c_e[0][0],
        )
    }
    Ok(())
}
