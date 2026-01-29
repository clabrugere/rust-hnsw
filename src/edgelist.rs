use std::collections::BTreeSet;

use super::candidate::Candidate;

// Utility struct to maintain a fixed capacity ordered set of candidates, popping the worst candidate when exceeding capacity
#[derive(Debug)]
pub(super) struct SortedEdgeList {
    pub(super) set: BTreeSet<Candidate>,
    capacity: usize,
}

impl SortedEdgeList {
    pub fn new(capacity: usize) -> Self {
        Self {
            set: BTreeSet::new(),
            capacity,
        }
    }

    pub fn insert(&mut self, candidate: Candidate) {
        // remove existing candidate with same id if the new one is closer
        if let Some(existing) = self.set.iter().find(|c| c.id == candidate.id).cloned() {
            if existing > candidate {
                self.set.remove(&existing);
            }
        }

        self.set.insert(candidate);

        // remove the worst (largest distance) if capacity exceeded
        if self.set.len() > self.capacity {
            if let Some(&worst) = self.set.iter().next_back() {
                self.set.remove(&worst);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.set.iter().map(|c| c.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_edge_list_insert() {
        let mut list = SortedEdgeList::new(3);
        list.insert(Candidate::new(1, 0.5));
        list.insert(Candidate::new(2, 0.3));
        list.insert(Candidate::new(3, 0.7));

        assert_eq!(list.set.len(), 3);
        let ids: Vec<usize> = list.iter().collect();
        assert_eq!(ids, vec![2, 1, 3]); // sorted by distance
    }

    #[test]
    fn test_sorted_edge_list_capacity() {
        let mut list = SortedEdgeList::new(2);
        list.insert(Candidate::new(1, 0.5));
        list.insert(Candidate::new(2, 0.3));
        list.insert(Candidate::new(3, 0.7));

        assert_eq!(list.set.len(), 2);
        let ids: Vec<usize> = list.iter().collect();
        assert_eq!(ids, vec![2, 1]); // worst candidate (3) removed
    }

    #[test]
    fn test_sorted_edge_list_update_closer() {
        let mut list = SortedEdgeList::new(3);
        list.insert(Candidate::new(1, 0.5));
        list.insert(Candidate::new(1, 0.3)); // closer distance for same id

        assert_eq!(list.set.len(), 1);
        let candidate = list.set.iter().next().unwrap();
        assert_eq!(candidate.distance, 0.3);
    }

    #[test]
    fn test_sorted_edge_list_empty() {
        let list = SortedEdgeList::new(5);
        assert_eq!(list.set.len(), 0);
        assert_eq!(list.iter().count(), 0);
    }

    #[test]
    fn test_sorted_edge_list_zero_capacity() {
        let mut list = SortedEdgeList::new(0);
        list.insert(Candidate::new(1, 0.5));

        assert_eq!(list.set.len(), 0);
    }
}
