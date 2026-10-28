use std::{
    fs::{create_dir_all, write},
    io::Error,
    path::Path,
};

use conspire::{
    constitutive::{
        solid::{
            elastic::doc as elastic, elastic_hyperviscous::doc as elastic_hyperviscous,
            elastic_viscoplastic::doc as elastic_viscoplastic, hyperelastic::doc as hyperelastic,
            hyperelastic_viscoplastic::doc as hyperelastic_viscoplastic,
            hyperviscoelastic::doc as hyperviscoelastic, thermoelastic::doc as thermoelastic,
            thermohyperelastic::doc as thermohyperelastic, viscoelastic::doc as viscoelastic,
        },
        thermal::conduction::doc as thermal_conduction,
    },
    math::integrate::doc::{
        EXPLICIT, IMPLICIT, backward_euler, bogacki_shampine, bogacki_shampine_fixed_step,
        dormand_prince, dormand_prince_fixed_step, euler, heun, midpoint, ralston, verner_8,
        verner_8_fixed_step, verner_9, verner_9_fixed_step,
    },
};

fn main() -> Result<(), Error> {
    math()?;
    constitutive()?;
    thermal()
}

fn math() -> Result<(), Error> {
    write_models(&[
        vec![["math/integrate/explicit", EXPLICIT]],
        bogacki_shampine(),
        bogacki_shampine_fixed_step(),
        dormand_prince(),
        dormand_prince_fixed_step(),
        euler(),
        heun(),
        midpoint(),
        ralston(),
        verner_8(),
        verner_8_fixed_step(),
        verner_9(),
        verner_9_fixed_step(),
        vec![["math/integrate/implicit", IMPLICIT]],
        backward_euler(),
    ])
}

fn constitutive() -> Result<(), Error> {
    write_models(&[
        vec![["constitutive/solid/elastic", elastic::DOC]],
        elastic::almansi_hamel_eulerian(),
        elastic::almansi_hamel_lagrangian(),
        elastic::bazant_itskov_eulerian(),
        elastic::bazant_itskov_lagrangian(),
        elastic::hencky(),
        elastic::saint_venant_kirchhoff(),
        elastic::seth_hill_eulerian(),
        elastic::seth_hill_lagrangian(),
        vec![["constitutive/solid/hyperelastic", hyperelastic::DOC]],
        hyperelastic::arruda_boyce(),
        hyperelastic::blatz_ko(),
        hyperelastic::carroll(),
        hyperelastic::fung(),
        hyperelastic::gent(),
        hyperelastic::hencky(),
        hyperelastic::isihara(),
        hyperelastic::mooney_rivlin(),
        hyperelastic::neo_hookean(),
        hyperelastic::ogden(),
        hyperelastic::saint_venant_kirchhoff(),
        hyperelastic::yeoh(),
        vec![["constitutive/solid/viscoelastic", viscoelastic::DOC]],
        vec![[
            "constitutive/solid/elastic_hyperviscous",
            elastic_hyperviscous::DOC,
        ]],
        vec![[
            "constitutive/solid/hyperviscoelastic",
            hyperviscoelastic::DOC,
        ]],
        vec![[
            "constitutive/solid/elastic_viscoplastic",
            elastic_viscoplastic::DOC,
        ]],
        vec![[
            "constitutive/solid/hyperelastic_viscoplastic",
            hyperelastic_viscoplastic::DOC,
        ]],
        vec![["constitutive/solid/thermoelastic", thermoelastic::DOC]],
        thermoelastic::almansi_hamel(),
        vec![[
            "constitutive/solid/thermohyperelastic",
            thermohyperelastic::DOC,
        ]],
        thermohyperelastic::saint_venant_kirchhoff(),
    ])
}

fn thermal() -> Result<(), Error> {
    write_models(&[
        vec![["constitutive/thermal/conduction", thermal_conduction::DOC]],
        thermal_conduction::fourier(),
    ])
}

fn write_models(models: &[Vec<[&str; 2]>]) -> Result<(), Error> {
    let mut path = "";
    models.iter().try_for_each(|model| {
        path = model[0][0];
        create_dir_all(Path::new(format!("target/doc/{path}").as_str()))?;
        write(
            Path::new(format!("target/doc/{path}/doc.md").as_str()),
            model[0][1]
                .replace("super::NeoHookean", "neo_hookean.html")
                .replace("super::ArrudaBoyce", "arruda_boyce.html")
                .replace("super::MooneyRivlin", "mooney_rivlin.html")
                .replace(
                    "super::super::hyperelastic::Hencky",
                    "../hyperelastic/hencky.html",
                )
                .replace(
                    "super::super::hyperelastic::SaintVenantKirchhoff",
                    "../hyperelastic/saint_venant_kirchhoff.html",
                )
                .replace("super::Hencky", "hencky.html")
                .replace("super::SaintVenantKirchhoff", "saint_venant_kirchhoff.html")
                .replace("super::AlmansiHamelEulerian", "almansi_hamel_eulerian.html")
                .replace(
                    "super::AlmansiHamelLagrangian",
                    "almansi_hamel_lagrangian.html",
                )
                .replace("super::BazantItskovEulerian", "bazant_itskov_eulerian.html")
                .replace(
                    "super::BazantItskovLagrangian",
                    "bazant_itskov_lagrangian.html",
                )
                .replace("super::SethHillEulerian", "seth_hill_eulerian.html")
                .replace("super::SethHillLagrangian", "seth_hill_lagrangian.html")
                .replace(
                    "[Bogacki-Shampine](`crate::math::integrate::BogackiShampine`)",
                    "[Bogacki-Shampine](bogacki_shampine.html)",
                )
                .replace(
                    "[Dormand-Prince](`crate::math::integrate::DormandPrince`)",
                    "[Dormand-Prince](dormand_prince.html)",
                )
                .replace(
                    "[Verner 8](`crate::math::integrate::Verner8`)",
                    "[Verner 8](verner_8.html)",
                )
                .replace(
                    "[Verner 9](`crate::math::integrate::Verner9`)",
                    "[Verner 9](verner_9.html)",
                ),
        )?;
        model.iter().skip(1).try_for_each(|[method, doc]| {
            if doc.is_empty() {
                write(
                    Path::new(format!("target/doc/{path}/{method}.md").as_str()),
                    "@private",
                )
            } else {
                write(
                    Path::new(format!("target/doc/{path}/{method}.md").as_str()),
                    doc,
                )
            }
        })
    })
}
