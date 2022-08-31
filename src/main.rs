#![warn(clippy::pedantic, clippy::nursery)]

use muzero::mcts::Mcts;
use muzero::tictactoe::TicTacToeEnv;

fn main() {
    let t = TicTacToeEnv::new();
    let m = Mcts::new(t);

    for _ in 0..1_000_000 {
        m.search(0, 1000.0);
    }

    println!("{}", m);
}
