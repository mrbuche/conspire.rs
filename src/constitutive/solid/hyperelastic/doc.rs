pub const DOC: &str = include_str!("doc.md");

pub fn arruda_boyce<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/arruda_boyce",
            include_str!("arruda_boyce/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("arruda_boyce/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("arruda_boyce/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("arruda_boyce/helmholtz_free_energy_density.md"),
        ],
    ]
}
