pub trait Split {
    fn split(self) -> Self;
}

impl Split for u8 {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for u16 {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for u32 {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for u64 {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for usize {
    fn split(self) -> Self {
        self / 2
    }
}
