pub const DOC: &str = include_str!("doc.md");

pub fn almansi_hamel<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic_hyperviscous/almansi_hamel",
            include_str!("almansi_hamel/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("almansi_hamel/cauchy_stress.md"),
        ],
        [
            "cauchy_rate_tangent_stiffness",
            include_str!("almansi_hamel/cauchy_rate_tangent_stiffness.md"),
        ],
        [
            "viscous_dissipation",
            include_str!("almansi_hamel/viscous_dissipation.md"),
        ],
    ]
}
