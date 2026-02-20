#![cfg(feature = "constitutive")]

use conspire::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::{
            ElasticMultiplicativeViscoplastic, ElasticViscoplasticAdditiveElastic,
            ElasticViscoplasticAdditiveViscoplastic,
        },
        solid::{
            elastic_plastic::ElasticPlasticOrViscoplastic,
            elastic_viscoplastic::AppliedLoad,
            hyperelastic::{Hencky, NeoHookean},
            hyperelastic_viscoplastic::SecondOrderMinimize,
        },
    },
    math::{Tensor, TestError, integrate::DormandPrince, optimize::NewtonRaphson},
};

#[test]
fn demonstrate() -> Result<(), TestError> {
    let model = ElasticViscoplasticAdditiveElastic::from((
        ElasticMultiplicativeViscoplastic::from((
            Hencky {
                bulk_modulus: 300.0,
                shear_modulus: 100.0,
            },
            ElasticViscoplasticAdditiveViscoplastic::from((
                ElasticMultiplicativeViscoplastic::from((
                    Hencky {
                        bulk_modulus: 0.0,
                        shear_modulus: 25.0,
                    },
                    ElasticViscoplasticAdditiveViscoplastic::from((
                        ElasticMultiplicativeViscoplastic::from((
                            Hencky {
                                bulk_modulus: 0.0,
                                shear_modulus: 10.0,
                            },
                            ViscoplasticFlow {
                                yield_stress: 3.0,
                                hardening_slope: 0.0,
                                rate_sensitivity: 0.25,
                                reference_flow_rate: 0.04,
                            },
                        )),
                        ViscoplasticFlow {
                            yield_stress: 2.0,
                            hardening_slope: 0.0,
                            rate_sensitivity: 0.25,
                            reference_flow_rate: 0.02,
                        },
                    )),
                )),
                ViscoplasticFlow {
                    yield_stress: 1.0,
                    hardening_slope: 1.0,
                    rate_sensitivity: 0.25,
                    reference_flow_rate: 0.01,
                },
            )),
        )),
        NeoHookean {
            bulk_modulus: 0.0,
            shear_modulus: 3.3,
        },
    ));
    let (t, f, f_p) = model.minimize(
        AppliedLoad::UniaxialStress(
            |t| {
                if t < 0.25_f64.exp() - 1.0 {
                    1.0 + t
                } else if t < 0.25_f64.exp() - 1.0 + 0.13_f64.exp() - 1.0 {
                    let t0 = 0.25_f64.exp() - 1.0;
                    0.25_f64.exp() - (t - t0)
                } else if t < 2.0 * (0.25_f64.exp() - 1.0) + 0.13_f64.exp() - 1.0 {
                    let t0 = 0.25_f64.exp() - 1.0 + 0.13_f64.exp() - 1.0 - 1.0;
                    0.25_f64.exp() - 0.13_f64.exp() + (t - t0)
                } else if t < 2.0 * (0.25_f64.exp() - 1.0) + (0.13_f64.exp() - 1.0) + (0.18_f64.exp() - 1.0) {
                    let t0 = 2.0 * (0.25_f64.exp() - 1.0) + 0.13_f64.exp() - 1.0;
                    2.0 * 0.25_f64.exp() - 0.13_f64.exp() - (t - t0)
                } else if t < 3.0 * (0.25_f64.exp() - 1.0) + (0.13_f64.exp() - 1.0) + (0.18_f64.exp() - 1.0) {
                    let t0 = 2.0 * (0.25_f64.exp() - 1.0) + (0.13_f64.exp() - 1.0) + (0.18_f64.exp() - 1.0) - 1.0;
                    2.0 * 0.25_f64.exp() - 0.13_f64.exp() - 0.18_f64.exp() + (t - t0)
                } else {
                    let t0 = 3.0 * (0.25_f64.exp() - 1.0) + (0.13_f64.exp() - 1.0) + (0.18_f64.exp() - 1.0);
                    3.0 * 0.25_f64.exp() - 0.13_f64.exp() - 0.18_f64.exp() - (t - t0)
                }
                // } else if t < 1.5 {
                //     2.0 - (t - 1.0)
                // } else if t < 2.5 {
                //     1.5 + (t - 1.5)
                // } else if t < 3.0 {
                //     2.5 - (t - 2.5)
                // } else {
                //     2.0 + (t - 3.0)
                // }
            },
            &[
                0.0,
                3.0 * (0.25_f64.exp() - 1.0) + 3.0 * (0.13_f64.exp() - 1.0),
            ],
        ),
        DormandPrince::default(),
        NewtonRaphson::default(),
    )?;
    for (t_i, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let (f_p_i, others) = s_i.into();
        let (others, eqps) = others.into();
        let (_, others) = others.into();
        let (others, eqps_1) = others.into();
        let (_, eqps_2) = others.into();
        let c_e = model.cauchy_stress(f_i, f_p_i)?;
        println!(
            "{}, {}, {}, {}, {}, {}",
            t_i,
            f_i[0][0].ln(),
            c_e[0][0],
            eqps,
            eqps_1,
            eqps_2,
        )
    }
    Ok(())
}
