use std::fmt::{Display, Formatter, Result};

#[derive(Debug)]
pub enum IndexError {
    EmptyIndex,
    EmptyLevel(usize),
    NodeNotFound(usize),
    NodeNotFoundInLevel { level_index: usize, node_id: usize },
    NoNeighborCandidates,
}

impl Display for IndexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EmptyIndex => write!(f, "Empty index"),
            Self::EmptyLevel(level) => write!(f, "Level {0} empty", level),
            Self::NodeNotFound(id) => write!(f, "Node {0} not found", id),
            Self::NodeNotFoundInLevel {
                level_index,
                node_id,
            } => {
                write!(
                    f,
                    "Node {0} not found in level {1} empty",
                    node_id, level_index
                )
            }
            Self::NoNeighborCandidates => write!(f, "Candidate not found"),
        }
    }
}
