pub const DOC: &str = include_str!("doc.md");

pub fn almansi_hamel<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic_viscoplastic/almansi_hamel",
            include_str!("almansi_hamel/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("almansi_hamel/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("almansi_hamel/cauchy_tangent_stiffness.md"),
        ],
    ]
}
