pub struct Orthants<const N: usize, U>([U; N]);

impl<const N: usize, U> From<[U; N]> for Orthants<N, U> {
    fn from(array: [U; N]) -> Self {
        Orthants(array)
    }
}
