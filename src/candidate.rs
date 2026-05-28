use std::cmp::Ordering;

/// Utility struct to be used with a binary heap in the neighbor search
#[derive(Debug, Copy, Clone)]
pub struct Candidate {
    pub id: usize,
    pub distance: f64,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance == other.distance
    }
}

impl Candidate {
    pub const fn new(id: usize, distance: f64) -> Self {
        Self { id, distance }
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // tie breaker on id to ensure deterministic ordering
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        let c1 = Candidate::new(1, 0.5);
        let c2 = Candidate::new(2, 0.3);
        let c3 = Candidate::new(3, 0.5);

        assert!(c2 < c1); // smaller distance comes first
        assert!(c1 < c3); // same distance, smaller id comes first
    }

    #[test]
    fn test_candidate_equality() {
        let c1 = Candidate::new(1, 0.5);
        let c2 = Candidate::new(1, 0.5);
        let c3 = Candidate::new(2, 0.5);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }
}
