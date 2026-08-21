pub const DOC: &str = include_str!("doc.md");

pub fn hencky<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic_viscoplastic/hencky",
            include_str!("hencky/doc.md"),
        ],
        ["cauchy_stress", include_str!("hencky/cauchy_stress.md")],
        [
            "cauchy_tangent_stiffness",
            include_str!("hencky/cauchy_tangent_stiffness.md"),
        ],
        [
            "helmholtz_free_energy_density",
            include_str!("hencky/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn saint_venant_kirchhoff<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic_viscoplastic/saint_venant_kirchhoff",
            include_str!("saint_venant_kirchhoff/doc.md"),
        ],
        [
            "second_piola_kirchhoff_stress",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
        [
            "helmholtz_free_energy_density",
            include_str!("saint_venant_kirchhoff/helmholtz_free_energy_density.md"),
        ],
    ]
}
