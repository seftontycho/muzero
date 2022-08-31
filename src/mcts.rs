use crate::envs::Environment;
use rand::prelude::*;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

pub struct MCTSNode {
    pub action: Option<usize>,
    pub children: Vec<usize>,
    pub n_visits: f64,
    pub q_value: f64,
}

impl Display for MCTSNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "action: {:?}, n_visits: {}, q_value: {}",
            self.action, self.n_visits, self.q_value
        )
    }
}

impl MCTSNode {
    pub const fn new(action: Option<usize>) -> Self {
        Self {
            action,
            children: Vec::new(),
            n_visits: 0.0,
            q_value: 0.0,
        }
    }

    pub fn ucb_score(&self, parent_n_visits: f64) -> f64 {
        if self.n_visits == 0.0 {
            return std::f64::MAX;
        }
        (self.q_value / self.n_visits) + (2.0 * (parent_n_visits).ln() / (self.n_visits)).sqrt()
    }
}

pub struct Mcts<T: Environment + Clone> {
    pub arena: Arc<Mutex<Vec<Mutex<MCTSNode>>>>,
    action_space: usize,
    env: T,
}

impl<T: Environment + Clone> Display for Mcts<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mcts {{").unwrap();
        writeln!(f, "    arena: {{").unwrap();

        let arena = self.arena.lock().unwrap();
        for (i, node) in arena.iter().take(10).enumerate() {
            writeln!(f, "        {}: {},", i, node.lock().unwrap()).unwrap();
        }
        writeln!(f, "    }},").unwrap();
        writeln!(f, "}}")
    }
}

impl<T: Environment + Clone> Mcts<T> {
    pub fn new(env: T) -> Self {
        let arena = Arc::new(Mutex::new(Vec::new()));
        let mut a = arena.lock().unwrap();
        a.push(Mutex::new(MCTSNode::new(None)));
        drop(a);

        let action_space = env.action_space();
        Self {
            arena,
            action_space,
            env,
        }
    }

    pub fn search(&self, root_index: usize, n_rollouts: f64) -> usize {
        let history = self.select_leaf(root_index);
        let result = self.rollout(&history, n_rollouts);
        let last = *history.last().unwrap();
        self.backpropagate(&history, result);
        last
    }

    fn backpropagate(&self, history: &[usize], result: f64) {
        let arena = self.arena.lock().unwrap();
        let offset = (history.len() + 1) % 2;

        //only update for the player which has won.
        //suggested from https://jyopari.github.io/MCTS.html
        for i in history.iter().skip(offset).step_by(2) {
            let mut node = arena[*i].lock().unwrap();
            node.n_visits += 1.0;
            node.q_value += result;
        }
    }

    fn follow_trajectory(&self, history: &[usize]) -> (Option<T>, f64) {
        let mut env = self.env.clone();
        let arena = self.arena.lock().unwrap();

        let action_history: Vec<usize> = history[1..]
            .iter()
            .map(|&i| arena[i].lock().unwrap().action.unwrap())
            .collect();

        drop(arena);

        let mut total_reward = 0.0;

        for action in action_history {
            let (_, _, reward, done) = env.step(action);
            total_reward += reward;
            if done {
                return (None, total_reward);
            }
        }
        (Some(env), total_reward)
    }

    fn rollout(&self, history: &[usize], n: f64) -> f64 {
        let (env, mut total_reward) = self.follow_trajectory(history);
        if env.is_none() {
            return total_reward;
        }
        let mut env = env.unwrap();
        let mut rng = thread_rng();

        for _ in 0..n as usize {
            loop {
                let action = rng.gen_range(0..self.action_space);
                let (_, _, reward, done) = env.step(action);
                total_reward += reward;
                if done {
                    break;
                }
            }
        }

        total_reward / n
    }

    fn select_leaf(&self, root_index: usize) -> Vec<usize> {
        // get lock of arena
        let arena = self.arena.lock().unwrap();
        let root = arena[root_index].lock().unwrap();

        let mut current_index = root_index;
        let mut current_node = root;

        let mut history = Vec::new();

        // traverse to leaf node
        while !current_node.children.is_empty() {
            history.push(current_index);
            let mut best_child = None;
            let mut best_ucb = std::f64::MIN;

            for child_index in &current_node.children {
                let child = arena[*child_index].lock().unwrap();
                let ucb = child.ucb_score(current_node.n_visits);

                if ucb >= best_ucb {
                    best_child = Some(*child_index);
                    best_ucb = ucb;
                }
            }

            current_index = best_child.unwrap();
            current_node = arena[current_index].lock().unwrap();
        }

        history.push(current_index);

        if current_node.n_visits <= 0.0 {
            return history;
        }

        // unlock arena
        drop(current_node);
        drop(arena);

        // create new children
        let mut new_children = Vec::new();
        for action in 0..self.action_space {
            let child = MCTSNode::new(Some(action));
            new_children.push(Mutex::new(child));
        }

        // get lock on arena
        let mut arena = self.arena.lock().unwrap();
        let index = arena.len();

        // add new children to arena
        arena.extend(new_children);
        arena[current_index]
            .lock()
            .unwrap()
            .children
            .extend((index..index + self.action_space).collect::<Vec<usize>>());

        history.push(index);
        history
    }
}

// TODO:
// - Add action masks
// - Add action priors
// - Add noise to action selection
