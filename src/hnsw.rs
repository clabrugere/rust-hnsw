use core::f64;
use rand::{Rng, rng, seq::IteratorRandom};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;

use super::candidate::Candidate;
use super::edgelist::SortedEdgeList;
use super::errors::{IndexError, IndexResult};

type Nodes<T, const D: usize> = HashMap<usize, [T; D]>;
type Level = HashMap<usize, SortedEdgeList>;

/// Utility struct to store a nearest neighbor search result
#[derive(Debug)]
pub struct SearchResult<'v, T, const D: usize> {
    pub vector: &'v [T; D],
    pub distance: f64,
}

impl<'c, T, const D: usize> SearchResult<'c, T, D> {
    pub const fn new(vector: &'c [T; D], distance: f64) -> Self {
        Self { vector, distance }
    }
}

pub struct HNSW<T, const D: usize, F> {
    connections: usize, // M parameter
    ef_construction: usize,
    distance_metric: F,
    max_connections: usize,   // Mmax parameter
    max_connections_0: usize, // Mmax0
    nodes: Nodes<T, D>,
    levels: Vec<Level>,
    next_id: usize,
}

impl<T, const D: usize, F> HNSW<T, D, F>
where
    T: Copy,
    F: Fn(&[T], &[T]) -> f64,
{
    pub fn new(connections: usize, ef_construction: usize, distance_metric: F) -> Self {
        Self {
            connections,
            ef_construction,
            distance_metric,
            // heuristic to bound the connectivity of the levels
            max_connections: (1.5 * connections as f32).round() as usize,
            max_connections_0: 2 * connections,
            nodes: Nodes::new(),
            levels: Vec::new(),
            next_id: 0,
        }
    }

    fn distance(&self, a: &[T; D], b: &[T; D]) -> f64 {
        (self.distance_metric)(a, b)
    }

    fn get_edgelist_mut(
        &mut self,
        level_index: usize,
        node_id: usize,
    ) -> IndexResult<&mut SortedEdgeList> {
        self.levels[level_index]
            .get_mut(&node_id)
            .ok_or(IndexError::NodeNotFoundInLevel {
                level_index,
                node_id,
            })
    }

    fn get_vector(&self, node_id: usize) -> IndexResult<&[T; D]> {
        self.nodes
            .get(&node_id)
            .ok_or(IndexError::NodeNotFound(node_id))
    }

    /// Define the highest level by sampling from an exponentially decaying distribution
    fn sample_max_level_index(&self) -> usize {
        let level_multiplier = 1.0 / (self.connections as f64).ln();
        let log_p = rng().random_range(f64::EPSILON..=1.0).ln();

        (-(log_p * level_multiplier).floor()).max(1.0) as usize - 1
    }

    fn insert_top_level(&mut self, id: usize) {
        let max_connections = self.get_max_connections(self.levels.len());
        let level = Level::from([(id, SortedEdgeList::new(max_connections))]);
        self.levels.push(level);
    }

    const fn get_max_connections(&self, level_index: usize) -> usize {
        if level_index > 0 {
            self.max_connections
        } else {
            self.max_connections_0
        }
    }

    /// Randomly sample a node in a given level. We are guaranteed to have at least one point when invoking this method
    fn sample_entry_id(&self, level_index: usize) -> IndexResult<usize> {
        self.levels[level_index]
            .keys()
            .choose(&mut rng())
            .copied()
            .ok_or(IndexError::EmptyLevel(level_index))
    }

    /// Insert a new vector in the index and return its unique id
    fn insert_vector(&mut self, vector: &[T; D]) -> usize {
        let id = self.next_id;
        self.nodes.insert(id, *vector);
        self.next_id += 1;

        id
    }

    fn select_neighbors(&self, candidates: &[Candidate], k: usize) -> IndexResult<Vec<Candidate>> {
        let mut result: Vec<Candidate> = Vec::with_capacity(k);
        let mut accepted_vectors: Vec<&[T; D]> = Vec::with_capacity(k);
        for &candidate in candidates {
            if result.len() >= k {
                break;
            }
            let candidate_vec = self.get_vector(candidate.id)?;
            if !self.is_dominated(candidate_vec, candidate.distance, &accepted_vectors) {
                result.push(candidate);
                accepted_vectors.push(candidate_vec);
            }
        }
        Ok(result)
    }

    // Returns true if the candidate is at least as close to some already-selected neighbor as it
    // is to the query — meaning it lies in a direction already covered, so adding it would cluster
    // neighbors rather than spread them out
    fn is_dominated(
        &self,
        candidate_vec: &[T; D],
        candidate_dist: f64,
        accepted_vectors: &[&[T; D]],
    ) -> bool {
        accepted_vectors
            .iter()
            .any(|&accepted_vec| candidate_dist >= self.distance(candidate_vec, accepted_vec))
    }

    /// Returns all the indices of neighboring nodes of a given node id and level index, if they exist
    fn get_neighbors(&self, level_index: usize, node_id: usize) -> Option<&SortedEdgeList> {
        self.levels[level_index].get(&node_id)
    }

    /// Create a bidirectional edge between a node id and a set of neighbors, in a given level
    /// Pop edge with largest distance if the edge list reached capacity (pruning step)
    fn connect_neighbors(
        &mut self,
        level_index: usize,
        node_id: usize,
        neighbors: &[Candidate],
    ) -> IndexResult<()> {
        for candidate in neighbors {
            self.get_edgelist_mut(level_index, node_id)?
                .insert(*candidate);
            self.get_edgelist_mut(level_index, candidate.id)?
                .insert(Candidate::new(node_id, candidate.distance));
        }
        Ok(())
    }

    /// Perform BFS in a level from a starting set of nodes, and return the nearest `ef` closest neighbors found
    fn search_level(
        &self,
        level_index: usize,
        query: &[T; D],
        entry_ids: &[usize],
        ef: usize,
    ) -> IndexResult<Vec<Candidate>> {
        if ef == 0 {
            return Ok(Vec::new());
        }

        let mut candidates = BinaryHeap::with_capacity(ef); // min heap
        let mut nearest_neighbors = BinaryHeap::with_capacity(ef); // max heap
        let mut visited = HashSet::with_capacity(ef * 2);

        for &entry_id in entry_ids {
            let distance = self.distance(query, self.get_vector(entry_id)?);
            let candidate = Candidate::new(entry_id, distance);
            candidates.push(Reverse(candidate));
            nearest_neighbors.push(candidate);
            visited.insert(entry_id);
        }

        while let Some(closest) = candidates.pop().map(|c| c.0) {
            let mut furthest_distance = nearest_neighbors
                .peek()
                .map_or(f64::INFINITY, |c| c.distance);

            // all closest neighbors have been explored
            if closest.distance > furthest_distance {
                break;
            }

            if let Some(neighbor_ids) = self.get_neighbors(level_index, closest.id) {
                let unvisited_neighbor_ids = neighbor_ids
                    .iter()
                    .filter(|&neighbor_id| visited.insert(neighbor_id));

                for neighbor_id in unvisited_neighbor_ids {
                    let distance = self.distance(query, self.get_vector(neighbor_id)?);

                    if nearest_neighbors.len() < ef || distance < furthest_distance {
                        let candidate = Candidate::new(neighbor_id, distance);
                        candidates.push(Reverse(candidate));
                        nearest_neighbors.push(candidate);

                        if nearest_neighbors.len() > ef {
                            nearest_neighbors.pop();
                            furthest_distance = nearest_neighbors
                                .peek()
                                .map_or(f64::INFINITY, |c| c.distance);
                        }
                    }
                }
            }
        }

        Ok(nearest_neighbors.into_sorted_vec())
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return the number of vectors stored in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of levels in the index
    pub const fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Insert a new vector in the index
    pub fn insert(&mut self, vector: &[T; D]) -> IndexResult<()> {
        let node_id = self.insert_vector(vector);

        if self.levels.is_empty() {
            self.insert_top_level(node_id);
            return Ok(());
        }

        let top_level_index = self.num_levels() - 1;
        let sampled_level = self.sample_max_level_index();

        // handle the case of sampling a level higher than the current top level
        let max_level_index = if sampled_level > top_level_index {
            self.insert_top_level(node_id);
            top_level_index
        } else {
            sampled_level
        };

        // sample entry point
        let mut entry_ids = vec![self.sample_entry_id(top_level_index)?];

        // travel hierarchy for levels above the highest level of this node
        for level_index in (max_level_index + 1..=top_level_index).rev() {
            entry_ids = self
                .search_level(level_index, vector, &entry_ids, 1)?
                .into_iter()
                .map(|candidate| candidate.id)
                .collect();
        }

        // travel hierarchy for levels equal or below the highest level of this node
        for level_index in (0..=max_level_index).rev() {
            // add the node to the level
            let max_connections = self.get_max_connections(level_index);
            self.levels[level_index].insert(node_id, SortedEdgeList::new(max_connections));

            // look for neighbors to connect
            let candidates =
                self.search_level(level_index, vector, &entry_ids, self.ef_construction)?;

            let neighbors = self.select_neighbors(&candidates, self.connections)?;
            self.connect_neighbors(level_index, node_id, &neighbors)?;
            entry_ids = candidates.iter().map(|c| c.id).collect();
        }
        Ok(())
    }

    /// Insert each element of an iterator in the index
    pub fn insert_batch(&mut self, batch: impl IntoIterator<Item = [T; D]>) -> IndexResult<()> {
        batch.into_iter().try_for_each(|v| self.insert(&v))
    }

    /// Search for the `k` nearest neighbors of `query`; `ef >= k` controls beam width at the base level
    pub fn search(
        &self,
        query: &[T; D],
        k: usize,
        ef: usize,
    ) -> IndexResult<Vec<SearchResult<'_, T, D>>> {
        // check for edge cases
        if self.is_empty() {
            return Err(IndexError::EmptyIndex);
        } else if k == 0 {
            return Ok(Vec::new());
        }

        // sample a random node in the top layer to start the search from
        let top_level_index = self.num_levels() - 1;
        let mut entry_ids = vec![self.sample_entry_id(top_level_index)?];

        // travel the hierarchy from top to bottom by finding the closest entry point for the next level
        // by construction, we are guaranteed that the node found is also present in all the lower levels
        for level_index in (1..self.num_levels()).rev() {
            entry_ids = self
                .search_level(level_index, query, &entry_ids, 1)?
                .into_iter()
                .map(|candidate| candidate.id)
                .collect();
        }

        // perform full search on the lowest level
        let nearest_neighbors = self
            .search_level(0, query, &entry_ids, ef.max(k))?
            .into_iter()
            .take(k)
            .map(|c| Ok(SearchResult::new(self.get_vector(c.id)?, c.distance)))
            .collect::<IndexResult<Vec<_>>>()?;

        Ok(nearest_neighbors)
    }

    /// Reset the index by deleting all the vectors and layers
    pub fn clear(&mut self) {
        self.levels = Vec::new();
        self.nodes = Nodes::new();
        self.next_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::euclidean;

    fn create_index() -> HNSW<f64, 3, for<'a, 'b> fn(&'a [f64], &'b [f64]) -> f64> {
        HNSW::new(8, 8, euclidean)
    }

    #[test]
    fn test_new() {
        let index = create_index();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.num_levels(), 0);
    }

    #[test]
    fn test_insert() {
        let mut index = create_index();

        let vector1 = [1., 2., 3.];
        let vector2 = [4., 5., 6.];
        let vector3 = [7., 8., 9.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
        assert!(index.nodes.values().any(|v| v == &vector1));
        assert!(index.nodes.values().any(|v| v == &vector2));
        assert!(index.nodes.values().any(|v| v == &vector3));
    }

    #[test]
    fn test_insert_iterator() {
        let mut index = create_index();
        let iterator = (0..3).map(|i| [i as f64; 3]);

        index.insert_batch(iterator).unwrap();

        assert!(!index.is_empty());
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_level_density_decay() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        // check that the number of nodes in levels is smaller the higher the level
        let structure_ok = index.levels.windows(2).all(|w| {
            let (layer_0, layer_1) = (&w[0], &w[1]);
            layer_0.len() >= layer_1.len()
        });

        assert!(structure_ok);
    }

    #[test]
    fn test_max_connections() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        let structure_ok = (0..index.num_levels()).all(|level_index| {
            let max = index.get_max_connections(level_index);
            index.levels[level_index]
                .values()
                .all(|edges| edges.len() <= max)
        });

        assert!(structure_ok);
    }

    #[test]
    fn test_search_empty() {
        let index = create_index();
        let vector = [1., 2., 3.];

        assert!(index.search(&vector, 1, 1).is_err());
    }

    #[test]
    fn test_search_exact() {
        let mut index = create_index();
        let vector = [1., 2., 3.];

        index.insert(&vector).unwrap();
        let result = index.search(&vector, 1, 1).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].vector, &vector);
        assert!(result[0].distance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_search_ordering() {
        let mut index = create_index();
        let vector1 = [1., 1., 1.]; // distance ~1.73 from origin
        let vector2 = [0., 0., 0.]; // distance 0 from origin
        let vector3 = [2., 2., 2.]; // distance ~3.46 from origin
        let vector4 = [0.5, 0.5, 0.5]; // distance ~0.87 from origin

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();
        index.insert(&vector4).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 4, 4).unwrap();

        assert_eq!(result.len(), 4);
        // Results should be ordered by distance (closest first)
        assert_eq!(result[0].vector, &vector2); // closest
        assert_eq!(result[1].vector, &vector4); // second closest
        assert_eq!(result[2].vector, &vector1); // third closest
        assert_eq!(result[3].vector, &vector3); // farthest

        // Verify distances are in ascending order
        for i in 1..result.len() {
            assert!(result[i - 1].distance <= result[i].distance);
        }
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let mut index = create_index();
        let vector1 = [1., 2., 3.];
        let vector2 = [4., 5., 6.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 10, 10).unwrap(); // k > index size

        assert_eq!(result.len(), 2); // Should return all available vectors
        assert!(result.iter().any(|r| r.vector == &vector1));
        assert!(result.iter().any(|r| r.vector == &vector2));
    }

    #[test]
    fn test_search_k_zero() {
        let mut index = create_index();
        let vector = [1., 2., 3.];
        index.insert(&vector).unwrap();

        let query = [0., 0., 0.];
        let result = index.search(&query, 0, 0).unwrap();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_search_with_duplicates() {
        let mut index = create_index();
        let vector1 = [1., 2., 3.];
        let vector2 = [1., 2., 3.]; // duplicate
        let vector3 = [4., 5., 6.];

        index.insert(&vector1).unwrap();
        index.insert(&vector2).unwrap();
        index.insert(&vector3).unwrap();

        let query = [1., 2., 3.];
        let result = index.search(&query, 3, 3).unwrap();

        assert_eq!(result.len(), 3);
        // First two results should have distance 0 (exact matches)
        assert!(result[0].distance.abs() < f64::EPSILON);
        assert!(result[1].distance.abs() < f64::EPSILON);
    }

    #[test]
    fn test_search_ef_clamped_to_k() {
        let mut index = create_index();
        index.insert_batch((0..5).map(|i| [i as f64; 3])).unwrap();

        let query = [0.; 3];
        let result = index.search(&query, 3, 1).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_search_ef_larger_than_k_returns_k() {
        let mut index = create_index();
        index.insert_batch((0..20).map(|i| [i as f64; 3])).unwrap();

        let query = [0.; 3];
        let result = index.search(&query, 3, 20).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_select_neighbors_heuristic_diversity() {
        let mut index = HNSW::new(2, 8, euclidean);

        for i in 0..5 {
            index.insert(&[0.01 * i as f64, 0., 0.]).unwrap();
        }
        let outlier = [10., 0., 0.];
        index.insert(&outlier).unwrap();

        let result = index.search(&[10., 0., 0.], 1, 4).unwrap();
        assert_eq!(result[0].vector, &outlier);
    }

    #[test]
    fn test_clear() {
        let mut index = create_index();
        index.insert_batch((0..10).map(|i| [i as f64; 3])).unwrap();

        assert_eq!(index.len(), 10);

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }
}
