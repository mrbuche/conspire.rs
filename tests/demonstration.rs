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
            hyperelastic::{ArrudaBoyce, Hencky},
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
                bulk_modulus: 100.0,
                shear_modulus: 10.0,
            },
            ElasticViscoplasticAdditiveViscoplastic::from((
                ElasticMultiplicativeViscoplastic::from((
                    Hencky {
                        bulk_modulus: 0.0,
                        shear_modulus: 0.0,
                    },
                    ElasticViscoplasticAdditiveViscoplastic::from((
                        ElasticMultiplicativeViscoplastic::from((
                            Hencky {
                                bulk_modulus: 0.0,
                                shear_modulus: 0.0,
                            },
                            ViscoplasticFlow {
                                yield_stress: 1.0,
                                hardening_slope: 1.0,
                                rate_sensitivity: 0.25,
                                reference_flow_rate: 0.001,
                            },
                        )),
                        ViscoplasticFlow {
                            yield_stress: 1.0,
                            hardening_slope: 1.0,
                            rate_sensitivity: 0.25,
                            reference_flow_rate: 0.001,
                        },
                    )),
                )),
                ViscoplasticFlow {
                    yield_stress: 1.0,
                    hardening_slope: 1.0,
                    rate_sensitivity: 0.25,
                    reference_flow_rate: 0.001,
                },
            )),
        )),
        ArrudaBoyce {
            bulk_modulus: 0.0,
            shear_modulus: 1.0,
            number_of_links: 23.0,
        },
    ));
    let (t, f, f_p) = model.minimize(
        AppliedLoad::UniaxialStress(
            |t| {
                if t < 1.0 {
                    1.0 + t
                } else if t < 1.5 {
                    2.0 - (t - 1.0)
                } else if t < 2.5 {
                    1.5 + (t - 1.5)
                } else if t < 3.0 {
                    2.5 - (t - 2.5)
                } else {
                    2.0 + (t - 3.0)
                }
            },
            &[0.0, 4.0],
        ),
        DormandPrince::default(),
        NewtonRaphson::default(),
    )?;
    for (t_i, (f_i, s_i)) in t.iter().zip(f.iter().zip(f_p.iter())) {
        let (f_p_i, others) = s_i.into();
        let (others, eqps) = others.into();
        let (_, others) = others.into();
        let (_, eqps_1) = others.into();
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
