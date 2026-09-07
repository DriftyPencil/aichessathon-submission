"""Chess-rule and learning-target regressions, independent of model strength."""

import time
import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

import agent
from az_model import (
    MIRROR_ACTION_INDICES,
    POLICY_SIZE,
    AlphaZeroLite,
    action_index,
    encode_board,
    mirror_policy,
    model_from_state,
)
from chess_encoding import action_index as onnx_action_index
from chess_encoding import encode_board as onnx_encode_board
from training.distill import split_games
from training.train import (
    Experience,
    SelfPlayGame,
    batched_search,
    finish_game,
    legal_action_indices,
    mask_policy_logits,
    train_epochs,
)
from training.widen import widen_model

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        agent._GAME_BOARD = None
        agent._GAME_ROOT = None
        agent._EVALUATIONS.clear()

    def tearDown(self) -> None:
        agent._GAME_BOARD = None
        agent._GAME_ROOT = None
        agent._EVALUATIONS.clear()

    def test_mate_is_found_even_when_policy_assigns_it_near_zero(self) -> None:
        board = chess.Board(MATE_IN_ONE)
        logits = torch.zeros(POLICY_SIZE)
        logits[action_index(board, chess.Move.from_uci("d1d8"))] = -100
        with patch.object(agent, "_predict", return_value=(logits, -0.8)):
            move = agent._mcts_move(board, time.perf_counter() - 1)
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_proven_loss_is_never_chosen_over_unknown_move(self) -> None:
        root = agent.SearchNode()
        lost = agent.SearchNode(0.9)
        lost.visits = 10000
        lost.proven = 1.0
        unknown = agent.SearchNode(0.1)
        root.children = {chess.Move.from_uci("e2e4"): lost, chess.Move.from_uci("d2d4"): unknown}
        self.assertEqual(agent._best_move(root).uci(), "d2d4")

    def test_opponent_reply_preserves_search_and_repetition_history(self) -> None:
        board = chess.Board()
        board.push_uci("g1f3")
        agent._GAME_BOARD = board
        root = agent.SearchNode()
        reply_node = agent.SearchNode()
        reply = chess.Move.from_uci("g8f6")
        root.children[reply] = reply_node
        agent._GAME_ROOT = root
        target = board.copy()
        target.push(reply)
        restored = agent._restore_history(target.fen())
        self.assertEqual(len(restored.move_stack), 2)
        self.assertIs(agent._GAME_ROOT, reply_node)

    def test_backup_uses_opposite_perspectives(self) -> None:
        nodes = [agent.SearchNode() for _ in range(3)]
        agent._backpropagate(nodes, 0.7)
        self.assertEqual([n.value() for n in nodes], [0.7, -0.7, 0.7])

    def test_proven_draw_is_preferred_to_estimated_loss(self) -> None:
        root = agent.SearchNode()
        root.visits = 100
        draw = agent.SearchNode(0.01)
        draw.proven = 0.0
        losing = agent.SearchNode(0.99)
        losing.visits = 100
        losing.value_sum = 80.0
        root.children = {chess.Move.from_uci("e2e4"): draw, chess.Move.from_uci("d2d4"): losing}
        self.assertIs(agent._select_child(root)[1], draw)

    def test_winning_position_still_explores_low_prior_moves(self) -> None:
        root = agent.SearchNode()
        root.visits = 128
        root.value_sum = 115.2
        explored = agent.SearchNode(0.98)
        explored.visits = 128
        explored.value_sum = -115.2
        unexplored = agent.SearchNode(0.02)
        root.children = {
            chess.Move.from_uci("e2e4"): explored,
            chess.Move.from_uci("d2d4"): unexplored,
        }
        self.assertIs(agent._select_child(root)[1], unexplored)

    def test_emergency_clock_returns_legal_move_without_inference(self) -> None:
        with patch.object(agent, "_predict", side_effect=AssertionError("inference")):
            move = agent.get_move(chess.STARTING_FEN, 1)
        self.assertIn(chess.Move.from_uci(move), chess.Board().legal_moves)

    def test_cache_matches_network_inputs_including_en_passant_and_clock(self) -> None:
        board = chess.Board()
        prediction = (torch.zeros((1, POLICY_SIZE)), torch.zeros(1))
        with patch.object(agent, "_INFERENCE", return_value=prediction) as inference:
            first = agent._predict(board)
            board.fullmove_number += 1
            self.assertIs(agent._predict(board), first)
            self.assertEqual(inference.call_count, 1)
            board.halfmove_clock += 1
            agent._predict(board)
            board.ep_square = chess.E3
            agent._predict(board)
            board.turn = chess.BLACK
            agent._predict(board)
            board.castling_rights = 0
            agent._predict(board)
            self.assertEqual(inference.call_count, 5)

    def test_cache_is_bounded_and_evicts_least_recent_entry(self) -> None:
        prediction = (torch.zeros((1, POLICY_SIZE)), torch.zeros(1))
        board = chess.Board()
        with (
            patch.object(agent, "MAX_CACHED_EVALUATIONS", 2),
            patch.object(
                agent,
                "_INFERENCE",
                return_value=prediction,
            ) as inference,
        ):
            agent._predict(board)
            board.push_uci("e2e4")
            agent._predict(board)
            board.push_uci("e7e5")
            agent._predict(board)
            self.assertEqual(len(agent._EVALUATIONS), 2)
            agent._predict(chess.Board())
            self.assertEqual(inference.call_count, 4)

    def test_batched_cache_deduplicates_and_owns_each_policy_row(self) -> None:
        first = chess.Board()
        second = first.copy()
        second.push_uci("e2e4")
        policies = torch.zeros((2, POLICY_SIZE))
        with patch.object(agent, "_INFERENCE", return_value=(policies, torch.zeros(2))) as model:
            predictions = agent._predict_many([first, second, first.copy()])
        self.assertEqual(model.call_count, 1)
        self.assertEqual(model.call_args.args[0].shape[0], 2)
        self.assertIs(predictions[0], predictions[2])
        self.assertEqual(predictions[0][0].untyped_storage().nbytes(), POLICY_SIZE * 4)
        self.assertNotEqual(
            predictions[0][0].untyped_storage().data_ptr(), policies.untyped_storage().data_ptr()
        )

    def test_batched_search_removes_virtual_visits(self) -> None:
        def predictions(boards: list[chess.Board]) -> list[tuple[torch.Tensor, float]]:
            return [(torch.zeros(POLICY_SIZE), 0.0) for _ in boards]

        with (
            patch.object(agent, "MAX_SIMULATIONS", 17),
            patch.object(agent, "_predict_many", side_effect=predictions),
        ):
            move = agent._mcts_move(chess.Board(), time.perf_counter() + 10)
        self.assertIn(move, chess.Board().legal_moves)
        root = agent._GAME_ROOT
        self.assertIsNotNone(root)
        assert root is not None
        self.assertEqual(root.visits, 17)
        self.assertEqual(sum(child.visits for child in root.children.values()), 17)
        self.assertAlmostEqual(root.value_sum, 0.0)
        pending = [root]
        counted = 0
        while pending:
            current = pending.pop()
            counted += 1
            pending.extend(current.children.values())
        self.assertEqual(root.size, counted)

    def test_virtual_visits_are_removed_when_batch_inference_fails(self) -> None:
        with (
            patch.object(agent, "_predict", return_value=(torch.zeros(POLICY_SIZE), 0.0)),
            patch.object(agent, "_predict_many", side_effect=RuntimeError("failed inference")),
            self.assertRaisesRegex(RuntimeError, "failed inference"),
        ):
            agent._mcts_move(chess.Board(), time.perf_counter() + 10)
        root = agent._GAME_ROOT
        assert root is not None
        self.assertEqual(root.visits, 0)
        self.assertEqual(root.value_sum, 0.0)
        self.assertEqual(sum(child.visits for child in root.children.values()), 0)


class LearningTests(unittest.TestCase):
    def test_torch_free_encoding_matches_training_encoding(self) -> None:
        board = chess.Board()
        rng = np.random.default_rng(77)
        for _ in range(160):
            np.testing.assert_array_equal(onnx_encode_board(board), encode_board(board))
            for move in board.legal_moves:
                self.assertEqual(onnx_action_index(board, move), action_index(board, move))
            if board.is_game_over(claim_draw=True):
                board.reset()
            moves = list(board.legal_moves)
            board.push(moves[int(rng.integers(len(moves)))])

    def test_vectorized_legal_loss_matches_ragged_loss_and_gradients(self) -> None:
        logits = torch.randn(3, POLICY_SIZE, requires_grad=True)
        actions = [np.asarray([1, 5]), np.asarray([0, 10, 17]), None]
        targets = torch.zeros_like(logits)
        targets[0, [1, 5]] = torch.tensor([0.3, 0.7])
        targets[1, [0, 10, 17]] = torch.tensor([0.2, 0.3, 0.5])
        targets[2] = 1 / POLICY_SIZE
        vectorized = -(targets * mask_policy_logits(logits, actions).log_softmax(1)).sum(1).mean()
        reference = torch.stack(
            [
                -(targets[row, indices] * logits[row, indices].log_softmax(0)).sum()
                for row, indices in enumerate(
                    [torch.tensor([1, 5]), torch.tensor([0, 10, 17]), torch.arange(POLICY_SIZE)]
                )
            ]
        ).mean()
        torch.testing.assert_close(vectorized, reference)
        torch.testing.assert_close(
            torch.autograd.grad(vectorized, logits, retain_graph=True)[0],
            torch.autograd.grad(reference, logits)[0],
        )

    def test_widening_preserves_outputs_and_checkpoint_architecture(self) -> None:
        source = AlphaZeroLite(channels=8, blocks=1).eval()
        target = widen_model(source, channels=12, blocks=2)
        restored = model_from_state(target.state_dict()).eval()
        inputs = torch.randn(4, 18, 8, 8)
        with torch.inference_mode():
            torch.testing.assert_close(target(inputs), source(inputs), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(restored(inputs), target(inputs))

    def test_policy_reflection_is_an_involution(self) -> None:
        policy = np.arange(POLICY_SIZE, dtype=np.float32)
        np.testing.assert_array_equal(mirror_policy(mirror_policy(policy)), policy)
        np.testing.assert_array_equal(
            MIRROR_ACTION_INDICES[MIRROR_ACTION_INDICES], np.arange(POLICY_SIZE)
        )

    def test_truncated_game_does_not_teach_a_draw(self) -> None:
        board = chess.Board()
        game = SelfPlayGame()
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        game.trajectory.append(
            (encode_board(board), policy, legal_action_indices(board), chess.WHITE)
        )
        self.assertEqual(finish_game(game, None, truncated=True)[0].value_weight, 0)
        self.assertEqual(finish_game(game, None)[0].value_weight, 1)
        self.assertEqual(finish_game(game, chess.BLACK)[0].value, -1)

    def test_castling_positions_are_not_file_reflected(self) -> None:
        board = chess.Board()
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[action_index(board, chess.Move.from_uci("e2e4"))] = 1
        example = Experience(encode_board(board), policy, 0, legal_action_indices(board))
        model = AlphaZeroLite(channels=8, blocks=1)
        optimizer = torch.optim.AdamW(model.parameters())
        with patch("training.train.mirror_state", side_effect=AssertionError("castling")):
            train_epochs(
                model, [example] * 4, optimizer, torch.device("cpu"), 1, 4, np.random.default_rng(0)
            )

    def test_selfplay_finds_exact_mate(self) -> None:
        board = chess.Board(MATE_IN_ONE)
        policies = batched_search(
            AlphaZeroLite(channels=8, blocks=1).eval(),
            [board],
            1,
            torch.device("cpu"),
            np.random.default_rng(0),
        )
        supported = [m for m in board.legal_moves if policies[0][action_index(board, m)] > 0]
        self.assertTrue(supported)
        for move in supported:
            child = board.copy()
            child.push(move)
            self.assertTrue(child.is_checkmate())

    def test_validation_holds_out_whole_games(self) -> None:
        examples = [
            Experience(np.zeros((18, 8, 8)), np.zeros(POLICY_SIZE), 0, game_id=i // 4)
            for i in range(40)
        ]
        training, validation = split_games(examples, 0.2, np.random.default_rng(0))
        self.assertEqual(len(training) + len(validation), len(examples))
        self.assertTrue({e.game_id for e in training}.isdisjoint(e.game_id for e in validation))


if __name__ == "__main__":
    unittest.main()
