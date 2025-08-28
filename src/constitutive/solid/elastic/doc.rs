pub const DOC: &str = include_str!("doc.md");

pub fn almansi_hamel<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/almansi_hamel",
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

pub fn hencky<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/hencky",
            include_str!("hencky/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("hencky/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("hencky/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
    ]
}
