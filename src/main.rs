#![warn(clippy::pedantic, clippy::nursery)]

use muzero::mcts::Mcts;
use muzero::tictactoe::TicTacToeEnv;

fn main() {
    let t = TicTacToeEnv::new();
    let m = Mcts::new(t);

    println!("{}", m);

    for _ in 0..1_000_000 {
        let _ = m.search(0, 1.0);
    }

    println!("{}", m);
    println!("{}", m.arena.lock().unwrap().len());
}
