use crate::episodes::Tensorable;
use std::fmt::Debug;

pub trait Environment {
    type Observation: Tensorable + Clone + Debug;

    fn step(&mut self, action: usize) -> (usize, Self::Observation, f64, bool);
    fn reset(&mut self) -> Self::Observation;
    fn action_space(&self) -> usize;
}
