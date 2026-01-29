use core::f64;
use rand::Rng;
use rand::{rng, seq::IteratorRandom};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;

use super::errors::IndexError;

/// Utility struct to be used with a binary heap in the neighbor search
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) struct Candidate {
    pub id: usize,
    pub distance: f64,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // tie breaker on id to ensure deterministic ordering
        match self
            .distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}

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

    fn insert(&mut self, candidate: Candidate) {
        // remove existing candidate with same id if the new one is closer
        if let Some(existing) = self.set.iter().find(|c| c.id == candidate.id).cloned() {
            if existing.distance > candidate.distance {
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

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.set.iter().map(|c| c.id)
    }
}

/// Utility struct to store a nearest neighbor search result
#[derive(Debug)]
pub struct SearchResult<'v, T, const D: usize> {
    pub vector: &'v [T; D],
    pub distance: f64,
}

type Nodes<T, const D: usize> = HashMap<usize, [T; D]>;
type Level = HashMap<usize, SortedEdgeList>;
type Candidates = Vec<Candidate>;

pub struct HNSW<T, const D: usize, F> {
    connections: usize, // M parameter
    ef_construction: usize,
    distance_metric: F,
    pub(crate) max_connections: usize,   // Mmax parameter
    pub(crate) max_connections_0: usize, // Mmax0
    pub(super) nodes: Nodes<T, D>,
    pub(super) levels: Vec<Level>,
    pub(super) next_id: usize,
}

impl<T, const D: usize, F> HNSW<T, D, F>
where
    T: Sized + Copy + Debug,
    F: Fn(&[T], &[T]) -> f64,
{
    pub fn new(connections: usize, ef_construction: usize, distance_metric: F) -> Self {
        // heuristic to bound the connectivity of the levels
        let max_connections = (1.5 * (connections as f32)).round() as usize;
        let max_connections_0 = 2 * connections;

        let nodes = Nodes::new();
        let levels = Vec::new();
        let next_id = 0;

        Self {
            connections,
            ef_construction,
            distance_metric,
            max_connections,
            max_connections_0,
            nodes,
            levels,
            next_id,
        }
    }

    fn distance(&self, a: &[T; D], b: &[T; D]) -> f64 {
        (self.distance_metric)(a, b)
    }

    fn get_edgelist_mut(
        &mut self,
        level_index: usize,
        node_id: usize,
    ) -> Result<&mut SortedEdgeList, IndexError> {
        self.levels[level_index]
            .get_mut(&node_id)
            .ok_or(IndexError::NodeNotFoundInLevel {
                level_index,
                node_id,
            })
    }

    fn get_vector(&self, node_id: usize) -> Result<&[T; D], IndexError> {
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

    fn insert_level_then_node(&mut self, id: usize, max_connections: usize) {
        let level = Level::from([(id, SortedEdgeList::new(max_connections))]);
        self.levels.push(level);
    }

    fn get_max_connections(&self, level_index: usize) -> usize {
        if level_index > 0 {
            self.max_connections
        } else {
            self.max_connections_0
        }
    }

    /// Randomly sample a node in the top layer. We are guaranteed to have at least one point when invoking this method
    fn sample_entry_id(&self, level_index: usize) -> Result<usize, IndexError> {
        self.levels[level_index]
            .keys()
            .choose(&mut rng())
            .cloned()
            .ok_or(IndexError::EmptyLevel(level_index))
    }

    /// Insert a new vector in the index and return its unique id
    fn insert_vector(&mut self, vector: &[T; D]) -> usize {
        let id = self.next_id;
        self.nodes.insert(id, *vector);
        self.next_id += 1;

        id
    }

    // TODO: implement heuristic as described in the paper
    fn select_neighbors<'c>(&self, candidates: &'c [Candidate], k: usize) -> &'c [Candidate] {
        if candidates.len() <= k {
            candidates
        } else {
            &candidates[..k]
        }
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
    ) -> Result<(), IndexError> {
        for candidate in neighbors {
            self.get_edgelist_mut(level_index, node_id)
                .map(|edge_list| edge_list.insert(*candidate))?;

            self.get_edgelist_mut(level_index, candidate.id)
                .map(|edge_list| {
                    edge_list.insert(Candidate {
                        id: node_id,
                        distance: candidate.distance,
                    })
                })?;
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
    ) -> Result<Candidates, IndexError> {
        if ef == 0 {
            return Ok(Vec::new());
        }

        let max_connections = self.get_max_connections(level_index);
        let mut candidates = BinaryHeap::with_capacity(max_connections); // min heap
        let mut nearest_neighbors = BinaryHeap::with_capacity(ef); // max heap
        let mut visited = HashSet::new();

        for &entry_id in entry_ids {
            let distance = self.distance(query, self.get_vector(entry_id)?);
            let candidate = Candidate {
                id: entry_id,
                distance,
            };
            candidates.push(Reverse(candidate));
            nearest_neighbors.push(candidate);
            visited.insert(entry_id);
        }

        while let Some(closest) = candidates.pop().map(|c| c.0) {
            let mut furthest_distance = nearest_neighbors
                .peek()
                .map(|c| c.distance)
                .ok_or(IndexError::NoNeighborCandidates)?;

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
                        let candidate = Candidate {
                            id: neighbor_id,
                            distance,
                        };
                        candidates.push(Reverse(candidate));
                        nearest_neighbors.push(candidate);

                        if nearest_neighbors.len() > ef {
                            nearest_neighbors.pop();
                        }
                        furthest_distance = nearest_neighbors
                            .peek()
                            .map(|c| c.distance)
                            .unwrap_or(f64::INFINITY);
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
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Insert a new vector in the index
    pub fn insert(&mut self, vector: &[T; D]) -> Result<(), IndexError> {
        let node_id = self.insert_vector(vector);

        if self.levels.is_empty() {
            self.insert_level_then_node(node_id, self.max_connections_0);
            return Ok(());
        }

        let top_level_index = self.num_levels() - 1;
        let mut max_level_index = self.sample_max_level_index();

        // handle the case of sampling a level higher than the current top level
        if max_level_index > top_level_index {
            self.insert_level_then_node(node_id, self.max_connections);
            max_level_index = top_level_index;
        }

        // sample entry point
        let entry_id = self.sample_entry_id(top_level_index)?;
        let mut entry_ids = Vec::from([entry_id]);

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

            let neighbors = self.select_neighbors(&candidates, self.connections);
            self.connect_neighbors(level_index, node_id, neighbors)?;
        }
        Ok(())
    }

    /// Insert each element of an iterator in the index
    pub fn insert_batch(
        &mut self,
        batch: impl IntoIterator<Item = [T; D]>,
    ) -> Result<(), IndexError> {
        for vector in batch {
            self.insert(&vector)?;
        }
        Ok(())
    }

    /// Search for the k nearest neighbors from the query vector by traveling the index
    pub fn search(
        &self,
        query: &[T; D],
        k: usize,
    ) -> Result<Vec<SearchResult<'_, T, D>>, IndexError> {
        // check for edge cases
        if self.is_empty() {
            return Err(IndexError::EmptyIndex);
        } else if k == 0 {
            return Ok(Vec::new());
        }

        // sample a random node in the top layer to start the search from
        let top_level_index = self.num_levels() - 1;
        let entry_id = self.sample_entry_id(top_level_index)?;
        let mut entry_ids = Vec::from([entry_id]);

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
            .search_level(0, query, &entry_ids, k)?
            .into_iter()
            .map(|c| {
                let result = SearchResult {
                    vector: self.get_vector(c.id)?,
                    distance: c.distance,
                };
                Ok(result)
            })
            .collect::<Result<Vec<_>, IndexError>>()?;

        Ok(nearest_neighbors)
    }

    /// Reset the index by deleting all the vectors and layers
    pub fn clear(&mut self) {
        self.levels = Vec::new();
        self.nodes = Nodes::new();
        self.next_id = 0;
    }
}
