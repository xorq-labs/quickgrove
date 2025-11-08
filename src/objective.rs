#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
    SquaredError,
    Logistic,
}

impl Objective {
    #[inline(always)]
    pub fn compute_score(&self, leaf_weight: f32) -> f32 {
        match self {
            Objective::SquaredError => leaf_weight,
            Objective::Logistic => 1.0 / (1.0 + (-leaf_weight).exp()),
        }
    }
}
