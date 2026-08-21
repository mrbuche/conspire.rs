pub const DOC: &str = include_str!("doc.md");

pub fn fourier<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/thermal/conduction/fourier",
            include_str!("fourier/doc.md"),
        ],
        ["potential", include_str!("fourier/potential.md")],
        ["heat_flux", include_str!("fourier/heat_flux.md")],
        [
            "heat_flux_tangent",
            include_str!("fourier/heat_flux_tangent.md"),
        ],
    ]
}
