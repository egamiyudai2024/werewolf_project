#main.py
import argparse
from lspo.trainer import LSPOTrainer
import config
from evaluation import run_evaluation
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="LSPO for Werewolf Game")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode to run the script in.')

    # --- 評価用の引数を追加 ---
    parser.add_argument('--eval_iter', type=int, default=1, 
                        help='Which agent iteration to evaluate (e.g., 1).') #help='Which agent iteration to evaluate.')
    parser.add_argument('--baseline_iter', type=int, default=0,
                        help='Which agent iteration to use as baseline (0 = base model).')
    parser.add_argument('--num_games', type=int, default=100, 
                        help='Number of games to run for evaluation.')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use (0 or 1).')
    parser.add_argument('--game_offset', type=int, default=0, help='Starting index for game logging (e.g., 0, 3, 6).')
    # --------------------------
    
    # 評価時に特定のモデルディレクトリ（アーカイブなど）を指定するための引数
    parser.add_argument('--model_dir', type=str, default=None, help='Path to specific model directory for evaluation (e.g., models_archive/0122_1430). If None, uses default models/.')

    #引数定義を追加
    parser.add_argument('--train_games', type=int, default=None, help='Override GAMES_PER_ITERATION_FOR_DATA')
    parser.add_argument('--candidates', type=int, default=None, help='Override CANDIDATE_ACTIONS_PER_TURN')
    parser.add_argument('--cfr_iter', type=int, default=None, help='Override CFR_TRAIN_ITERATIONS')
    parser.add_argument('--dpo_epochs', type=int, default=None, help='Override DPO_EPOCHS')

    args = parser.parse_args() #実際にコマンドラインから与えられた因数を解析

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 引数が指定されていれば、config.py の値を上書きする
    if args.train_games is not None:
        config.GAMES_PER_ITERATION_FOR_DATA = args.train_games
        print(f"[Override] GAMES_PER_ITERATION_FOR_DATA set to {config.GAMES_PER_ITERATION_FOR_DATA}")
    
    if args.candidates is not None:
        config.CANDIDATE_ACTIONS_PER_TURN = args.candidates
        print(f"[Override] CANDIDATE_ACTIONS_PER_TURN set to {config.CANDIDATE_ACTIONS_PER_TURN}")

    if args.cfr_iter is not None:
        config.CFR_TRAIN_ITERATIONS = args.cfr_iter
        print(f"[Override] CFR_TRAIN_ITERATIONS set to {config.CFR_TRAIN_ITERATIONS}")

    if args.dpo_epochs is not None:
        config.DPO_EPOCHS = args.dpo_epochs
        print(f"[Override] DPO_EPOCHS set to {config.DPO_EPOCHS}")

    if args.mode == 'train':
        #print("Starting LSPO training process...")
        #trainer = LSPOTrainer(config)
        #trainer.train()
        print(f"Starting LSPO training on GPU {args.gpu_id}, Game Offset {args.game_offset}...")
        trainer = LSPOTrainer(config)
        
        # ★ここで offset を渡す
        trainer.train(start_game_idx=args.game_offset)
        print("Training finished.")
        
    elif args.mode == 'evaluate':
        print(f"Starting evaluation process for Iteration {args.eval_iter} vs Iteration {args.baseline_iter}...")
        
        # 評価関数を呼び出す
        run_evaluation(
            lspo_iteration=args.eval_iter,
            baseline_iteration=args.baseline_iter,
            num_games=args.num_games,
            cfg=config,
            model_dir=args.model_dir
        )
        print("Evaluation finished.")


if __name__ == "__main__":
    main()