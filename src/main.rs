use mcts::MCTS;
use tictactoe::TicTacToeEnv;

mod envs;
mod episodes;
mod mcts;
mod tictactoe;

fn main() {
    let t = TicTacToeEnv::new();
    let m = MCTS::new(t);

    println!("{}", m);

    for _ in 0..1000000 {
        let _ = m.search(0, 1);
    }

    println!("{}", m);
    println!("{}", m.arena.lock().unwrap().len());
}
