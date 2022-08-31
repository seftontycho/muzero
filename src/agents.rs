pub trait Agent {
    fn get_action(&mut self, observation: &Tensor) -> i64;
}

struct MuzeroAgent {
    representation: MuZeroRepresentationNetwork,
    dynamics: MuZeroDynamicsNetwork,
    policy: MuZeroPolicyNetwork,
    config: MuZeroConfig,
    search_tree: MonteCarloSearchTree,
}

impl MuzeroAgent {
    fn new(config: MuZeroConfig) -> Self {
        let representation = MuZeroRepresentationNetwork::new(config.representation);
        let dynamics = MuZeroDynamicsNetwork::new(config.dynamics);
        let policy = MuZeroPolicyNetwork::new(config.policy);
        let search_tree = MonteCarloSearchTree::new(config.search_tree);

        MuzeroAgent {
            representation,
            dynamics,
            policy,
            config,
            search_tree,
        }
    }
}
