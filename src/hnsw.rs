use rand::{seq::IteratorRandom, Rng};
use std::{
    cmp::{min, Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    fmt::Debug,
};

type Nodes<T, const D: usize> = HashMap<usize, [T; D]>;
type Level = HashMap<usize, Vec<usize>>;

/// Utility struct to be used with a binary heap in the neighbor search
#[derive(Debug, PartialEq)]
struct Candidate {
    pub id: usize,
    pub distance: f64,
}

impl Candidate {
    pub fn new(id: usize, distance: f64) -> Self {
        Self { id, distance }
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

/// Utility struct to store a nearest neighbor search result
#[derive(Debug)]
pub struct SearchResult<'v, T, const D: usize> {
    pub vector: &'v [T; D],
    pub distance: f64,
}

impl<'v, T, const D: usize> SearchResult<'v, T, D> {
    pub fn new(vector: &'v [T; D], distance: f64) -> Self {
        Self { vector, distance }
    }
}

pub struct HNSW<T, const D: usize, F, R> {
    connections: usize, // M parameter
    ef_construction: usize,
    distance_metric: F,
    rng: R,
    pub(crate) max_connections: usize,   // Mmax parameter
    pub(crate) max_connections_0: usize, // Mmax0
    pub(super) nodes: Nodes<T, D>,
    pub(super) levels: Vec<Level>,
    pub(super) next_id: usize,
}

impl<T, const D: usize, F, R> HNSW<T, D, F, R>
where
    T: Sized + Copy + Debug,
    F: Fn(&[T], &[T]) -> f64,
    R: Rng,
{
    pub fn new(connections: usize, ef_construction: usize, distance_metric: F, rng: R) -> Self {
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
            rng,
            max_connections,
            max_connections_0,
            nodes,
            levels,
            next_id,
        }
    }

    /// Define the highest level by sampling from an exponentially decaying distribution
    fn sample_max_level_index(&mut self) -> usize {
        let level_multiplier = 1.0 / (self.connections as f64).ln();
        let log_p = self.rng.gen_range::<f64, _>(f64::EPSILON..=1.0).ln();

        -(log_p * level_multiplier).floor() as usize - 1
    }

    /// Randomly sample a node in the top layer. We are guaranteed to have at least one point when invoking this method
    fn sample_entry_id(&mut self, level_index: usize) -> usize {
        *self.levels[level_index]
            .keys()
            .choose(&mut self.rng)
            .unwrap()
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
        &candidates[..=min(k, candidates.len() - 1)]
    }

    /// Returns all the indices of neighboring nodes of a given node id and level index, if they exist
    fn get_neighbors(&self, level_index: usize, node_id: usize) -> Option<&Vec<usize>> {
        self.levels[level_index].get(&node_id)
    }

    /// Create a bidirectional edge between a node id and a set of neighbors, in a given level
    fn connect_neighbors(&mut self, level_index: usize, node_id: usize, neighbors: &[Candidate]) {
        for &Candidate { id, .. } in neighbors {
            self.levels[level_index].get_mut(&node_id).unwrap().push(id);
            self.levels[level_index].get_mut(&id).unwrap().push(node_id);
        }
    }

    /// Perform BFS in a level from a starting set of nodes, and return the nearest `ef` closest neighbors found
    fn search_level(
        &self,
        level_index: usize,
        query: &[T; D],
        entry_ids: &[usize],
        ef: usize,
    ) -> Vec<Candidate> {
        let max_connections = self.get_max_connections(level_index);
        let mut candidates = BinaryHeap::with_capacity(max_connections); // min heap
        let mut nearest_neighbors = BinaryHeap::with_capacity(ef); // max heap
        let mut visited = HashSet::new();

        for &entry_id in entry_ids {
            let distance = (self.distance_metric)(query, self.nodes.get(&entry_id).unwrap());

            visited.insert(entry_id);
            candidates.push(Reverse(Candidate::new(entry_id, distance)));
            nearest_neighbors.push(Candidate::new(entry_id, distance));
        }

        while let Some(closest) = candidates.pop().map(|c| c.0) {
            let furthest_distance = nearest_neighbors.peek().map(|c| c.distance).unwrap();

            // all closest neighbors have been explored
            if closest.distance > furthest_distance {
                break;
            }

            if let Some(neighbor_ids) = self.get_neighbors(level_index, closest.id) {
                neighbor_ids
                    .iter()
                    .filter(|&&neighbor_id| visited.insert(neighbor_id))
                    .for_each(|&neighbor_id| {
                        let distance =
                            (self.distance_metric)(query, self.nodes.get(&neighbor_id).unwrap());

                        if nearest_neighbors.len() < ef || distance < furthest_distance {
                            candidates.push(Reverse(Candidate::new(neighbor_id, distance)));
                            nearest_neighbors.push(Candidate::new(neighbor_id, distance));

                            if nearest_neighbors.len() > ef {
                                nearest_neighbors.pop();
                            }
                        }
                    })
            }
        }

        nearest_neighbors.into_sorted_vec()
    }

    fn insert_level_then_node(&mut self, id: usize, max_connections: usize) {
        let level = Level::from([(id, Vec::with_capacity(max_connections))]);
        self.levels.push(level);
    }

    fn get_max_connections(&self, level_index: usize) -> usize {
        if level_index > 0 {
            self.max_connections
        } else {
            self.max_connections_0
        }
    }

    fn prune_connections(&mut self, level_index: usize, neighbors: &[Candidate]) {
        // special case for the base level as described in the paper, they recommend to set it to 2M
        let max_connections = self.get_max_connections(level_index);

        for Candidate { id, .. } in neighbors {
            if let Some(edges) = self.levels[level_index].get_mut(id) {
                if edges.len() > max_connections {
                    // sort edges by the distances to node `id`
                    let query = self.nodes.get(id).unwrap();
                    let distances = edges
                        .iter()
                        .map(|&neighbor_id| {
                            Reverse(Candidate::new(
                                neighbor_id,
                                (self.distance_metric)(
                                    query,
                                    self.nodes.get(&neighbor_id).unwrap(),
                                ),
                            ))
                        })
                        .collect::<BinaryHeap<_>>()
                        .into_sorted_vec();

                    // prune connections to farthest nodes keeping only the `max_connections` closest
                    edges.clear();
                    edges.extend(distances.iter().take(max_connections).map(|c| c.0.id));
                }
            }
        }
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
    pub fn insert(&mut self, vector: &[T; D]) {
        let node_id = self.insert_vector(vector);

        if self.levels.is_empty() {
            self.insert_level_then_node(node_id, self.max_connections_0);
        } else {
            let top_level_index = self.num_levels() - 1;
            let mut max_level_index = self.sample_max_level_index();

            // handle the case of sampling a level higher than the current top level
            if max_level_index > top_level_index {
                self.insert_level_then_node(node_id, self.max_connections);
                max_level_index = top_level_index;
            }

            // sample entry point
            let mut entry_ids = Vec::from([self.sample_entry_id(top_level_index)]);

            // travel hierarchy for levels above the highest level of this node
            for level_index in (max_level_index + 1..=top_level_index).rev() {
                entry_ids = self
                    .search_level(level_index, vector, &entry_ids, 1)
                    .into_iter()
                    .map(|candidate| candidate.id)
                    .collect();
            }

            // travel hierarchy for levels equal or below the highest level of this node
            for level_index in (0..=max_level_index).rev() {
                // add the node to the level
                let max_connections = self.get_max_connections(level_index);
                self.levels[level_index].insert(node_id, Vec::with_capacity(max_connections));

                // look for neighbors to connect
                let candidates =
                    self.search_level(level_index, vector, &entry_ids, self.ef_construction);

                let neighbors = self.select_neighbors(&candidates, self.connections);
                self.connect_neighbors(level_index, node_id, neighbors);
                self.prune_connections(level_index, neighbors);
            }
        }
    }

    /// Insert each element of an iterator in the index
    pub fn insert_batch<I: Iterator<Item = [T; D]>>(&mut self, batch: I) {
        batch.for_each(|ref vector| self.insert(vector));
    }

    /// Search for the k nearest neighbors from the query vector by traveling the index
    pub fn search(
        &mut self,
        query: &[T; D],
        k: usize,
    ) -> Result<Vec<SearchResult<'_, T, D>>, &'static str> {
        if self.is_empty() {
            Err("index is empty")
        } else {
            // sample a random node in the top layer to start the search from
            let top_level_index = self.num_levels() - 1;
            let mut entry_ids = Vec::from([self.sample_entry_id(top_level_index)]);

            // travel the hierarchy from top to bottom by finding the closest entry point for the next level
            // by construction, we are guaranteed that the node found is also present in all the lower levels
            for level_index in (1..self.num_levels()).rev() {
                entry_ids = self
                    .search_level(level_index, query, &entry_ids, 1)
                    .into_iter()
                    .map(|candidate| candidate.id)
                    .collect();
            }

            // perform full search on the lowest level
            let nearest_neighbors = self
                .search_level(0, query, &entry_ids, k)
                .into_iter()
                .map(|c| SearchResult::new(self.nodes.get(&c.id).unwrap(), c.distance))
                .collect();

            Ok(nearest_neighbors)
        }
    }

    /// Reset the index by deleting all the vectors and layers
    pub fn clear(&mut self) {
        self.levels = Vec::new();
        self.nodes = Nodes::new();
        self.next_id = 0;
    }
}
