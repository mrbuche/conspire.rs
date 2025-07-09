pub const DOC: &str = include_str!("doc.md");
pub fn almansi_hamel<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "src/constitutive/solid/elastic/almansi_hamel",
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
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
    ]
}
