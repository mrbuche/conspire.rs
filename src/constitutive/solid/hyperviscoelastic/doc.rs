pub const DOC: &str = include_str!("doc.md");

pub fn saint_venant_kirchhoff<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperviscoelastic/saint_venant_kirchhoff",
            include_str!("saint_venant_kirchhoff/doc.md"),
        ],
        [
            "second_piola_kirchhoff_stress",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_rate_tangent_stiffness",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_rate_tangent_stiffness.md"),
        ],
        [
            "viscous_dissipation",
            include_str!("saint_venant_kirchhoff/viscous_dissipation.md"),
        ],
        [
            "helmholtz_free_energy_density",
            include_str!("saint_venant_kirchhoff/helmholtz_free_energy_density.md"),
        ],
    ]
}
