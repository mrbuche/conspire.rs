pub const DOC: &str = include_str!("doc.md");

pub fn almansi_hamel_eulerian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic_viscoplastic/almansi_hamel_eulerian",
            include_str!("almansi_hamel_eulerian/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("almansi_hamel_eulerian/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("almansi_hamel_eulerian/cauchy_tangent_stiffness.md"),
        ],
    ]
}
