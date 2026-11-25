"""Tests for the prefix/fill arms, the embedding loader and the starter integration.

Run (from hw-module-2-2/):  python -m unittest discover -s tests -v
Unit tests use synthetic data. Integration tests read the FAR-Trans CSVs (skipped if
absent); they use the Kaggle embeddings when present and hash stand-ins otherwise.
No test loads or downloads a model.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from common import DATA, DEMO_CUTOFF, EMBED_DIR, STARTER_PATH, text_sha  # noqa: E402
import embeddings  # noqa: E402
import run_demo  # noqa: E402
from fill import (DenseSpace, SparseSpace, oracle_fill, popularity_fill, prefix_of,  # noqa: E402
                  score_fill, shortlist_fill, tfidf_vectors, user_lists)

HAVE_DATA = (DATA / 'transactions.csv').exists()
HAVE_EMBEDDINGS = (EMBED_DIR / 'manifest.json').exists()

ELIGIBLE = [f'A{i:02d}' for i in range(20)]
POP_ORDER = list(reversed(ELIGIBLE))                 # A19 most popular
POP_RANK = {isin: r for r, isin in enumerate(POP_ORDER)}


class FillUnits(unittest.TestCase):
    def test_prefix_and_popularity_fill_rebuild_latest_buy(self):
        history = {'A03': 2, 'A07': 1}
        latest_buy = ['A07', 'A03'] + [i for i in POP_ORDER if i not in ('A07', 'A03')][:8]
        prefix = prefix_of(latest_buy, history)
        self.assertEqual(prefix, ['A07', 'A03'])
        self.assertEqual(popularity_fill(prefix, POP_ORDER), latest_buy)

    def test_score_fill_takes_best_unseen_and_breaks_ties_by_popularity(self):
        scores = {'A01': 0.9, 'A02': 0.9, 'A05': 0.8, 'A07': 1.0}
        ranking = score_fill(['A07'], ELIGIBLE, scores, POP_RANK)
        self.assertEqual(ranking[:4], ['A07', 'A02', 'A01', 'A05'])   # A02 more popular than A01
        self.assertEqual(len(set(ranking)), 10)
        self.assertNotIn('A07', ranking[1:])

    def test_full_prefix_leaves_no_slot_to_fill(self):
        prefix = ELIGIBLE[:10]
        self.assertEqual(score_fill(prefix, ELIGIBLE, {'A15': 1.0}, POP_RANK), prefix)
        self.assertEqual(shortlist_fill(prefix, POP_ORDER, {'A15': 1.0}, POP_RANK, 20), prefix)

    def test_shortlist_stays_inside_the_popularity_prior(self):
        scores = {'A00': 1.0, 'A18': 0.5}
        ranking = shortlist_fill(['A10'], POP_ORDER, scores, POP_RANK, n=9)
        self.assertNotIn('A00', ranking)                             # rank 19 > shortlist of 9
        self.assertEqual(ranking[1], 'A18')

    def test_dense_space_is_cosine_to_count_weighted_profile(self):
        vectors = {'A00': [1, 0], 'A01': [0, 1], 'A02': [1, 1], 'A03': [3, 1]}
        scores = DenseSpace(vectors).scores({'A00': 3, 'A01': 1})
        profile = np.array([3, 1]) / np.linalg.norm([3, 1])
        self.assertAlmostEqual(scores['A02'], float(np.dot([1, 1], profile) / np.sqrt(2)))
        self.assertAlmostEqual(scores['A03'], 1.0)

    def test_dense_space_rejects_broken_vectors(self):
        with self.assertRaises(ValueError):
            DenseSpace({'A00': [np.nan, 1.0]})
        with self.assertRaises(ValueError):
            DenseSpace({'A00': [0.0, 0.0]})

    def test_empty_profile_falls_back_to_popularity_and_is_reported(self):
        space = DenseSpace({i: [1.0, float(n)] for n, i in enumerate(ELIGIBLE)})
        latest = ['ZZZ'] + POP_ORDER[:9]
        lists, fallbacks = user_lists(latest, {'ZZZ': 1}, ELIGIBLE, POP_ORDER, POP_RANK,
                                      {'llm_fill': space})
        self.assertEqual(fallbacks, ['llm_fill'])
        self.assertEqual(lists['llm_fill'], lists['pop_fill'])

    def test_sparse_space_rejects_negative_weights(self):
        space = SparseSpace(tfidf_vectors({'A00': 'alpha beta', 'A01': 'beta gamma'}))
        with self.assertRaises(ValueError):
            space.scores({'A00': -1})

    def test_oracle_reads_only_new_labels(self):
        ranking = oracle_fill(['A03'], {'A11', 'A12'}, POP_ORDER)
        self.assertEqual(ranking[:3], ['A03', 'A11', 'A12'])
        self.assertEqual(len(set(ranking)), 10)


class EmbeddingLoader(unittest.TestCase):
    def _write(self, folder, texts, matrix, shas=None):
        (folder / 'texts.json').write_text(json.dumps(
            {'texts': [{'sha256': text_sha(t), 'text': t} for t in texts]}), encoding='utf-8')
        (folder / 'manifest.json').write_text(json.dumps(
            {'texts_sha256': shas or [text_sha(t) for t in texts]}), encoding='utf-8')
        np.save(folder / 'qwen3_embedding_0.6b.npy', np.asarray(matrix, dtype=np.float32))

    def test_alignment_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write(folder, ['a | Stock', 'b | Bond'], [[1, 0], [0, 1]])
            tables, _ = embeddings.load_tables(folder, folder / 'texts.json')
            row = {'assetName': 'b', 'assetCategory': 'Bond'}
            self.assertEqual(embeddings.vectors_for({'X': row}, tables['llm_fill'])['X'].tolist(), [0, 1])
            with self.assertRaises(ValueError):
                embeddings.vectors_for({'Y': {'assetName': 'c'}}, tables['llm_fill'])

    def test_misaligned_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write(folder, ['a', 'b'], [[1, 0], [0, 1]], shas=[text_sha('b'), text_sha('a')])
            with self.assertRaises(ValueError):
                embeddings.load_tables(folder, folder / 'texts.json')

    def test_non_finite_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._write(folder, ['a', 'b'], [[1, np.inf], [0, 1]])
            with self.assertRaises(ValueError):
                embeddings.load_tables(folder, folder / 'texts.json')


def demo_args(data_dir, output_dir, fake):
    return argparse.Namespace(data_dir=data_dir, output_dir=output_dir, baseline_dir=output_dir,
                              embed_dir=EMBED_DIR, cutoff=DEMO_CUTOFF, horizon_days=30,
                              max_users=100, fake_embeddings=fake)


@unittest.skipUnless(HAVE_DATA, 'FAR-Trans CSVs not found')
class StarterIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_demo.build(demo_args(DATA, None, not HAVE_EMBEDDINGS))

    def test_pop_fill_is_exactly_latest_buy(self):
        self.assertEqual(self.result['pop_fill_mismatch_users'], [])

    def test_every_list_keeps_the_prefix_and_is_valid(self):
        for uid in self.result['users']:
            prefix = self.result['predictions'][uid]['latest_buy']
            for method, ranking in self.result['predictions'][uid].items():
                self.assertEqual(len(set(ranking)), 10, method)
                if method.endswith('_fill') or method.endswith('_top20'):
                    k = next(r['free_slots'] for r in self.result['per_user'] if r['user_id'] == uid)
                    self.assertEqual(ranking[:10 - k], prefix[:10 - k], method)

    def test_fill_cannot_change_users_whose_labels_are_all_repeats(self):
        scores = {}
        for r in self.result['per_user']:
            if r['relevant_count'] and not r['new_to_user_count'] and (
                    r['method'].endswith('_fill') or r['method'].endswith('_top20')):
                scores.setdefault(r['user_id'], set()).add(round(r['ndcg_at_10'], 12))
        self.assertTrue(scores)
        self.assertTrue(all(len(v) == 1 for v in scores.values()))

    def test_protocol_and_baselines_match_a_fresh_starter_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'baseline'
            subprocess.run([sys.executable, str(STARTER_PATH), 'run', '--data-dir', str(DATA),
                            '--output-dir', str(out)], check=True, capture_output=True)
            check = run_demo.cross_check(self.result, out)
        self.assertTrue(check['protocol_hash_equal'])
        self.assertTrue(check['baseline_prediction_rows_equal'])
        self.assertTrue(check['baseline_per_user_ndcg_equal'])

    def test_prefix_and_fill_effects_add_up(self):
        """Team-merge identity: nDCG(A+B) = nDCG(A+pop) + nDCG(recency+B) - nDCG(recency+pop)."""
        r, rng = self.result, random.Random(0)
        ndcg = r['starter'].ndcg10
        for uid in r['users']:
            hist, rel, pred = r['history'][uid], r['relevant'][uid], r['predictions'][uid]
            p = min(10, len(hist))
            other = sorted(hist)
            rng.shuffle(other)                      # another prefix order (and choice, if > 10)
            other = other[:p]
            alt = ndcg(other + pred['pop_fill'][p:], rel)
            for fill in ('llm_fill', 'tfidf_fill', 'tfidf_top20'):
                both = ndcg(other + pred[fill][p:], rel)
                self.assertAlmostEqual(both, alt + ndcg(pred[fill], rel) - ndcg(pred['pop_fill'], rel),
                                       places=12)

    def test_lists_do_not_depend_on_the_python_hash_seed(self):
        code = ('import sys, json, hashlib, argparse; sys.path.insert(0, "src"); import run_demo; '
                f'from pathlib import Path; fake = {not HAVE_EMBEDDINGS}; '
                'a = argparse.Namespace(data_dir=Path(sys.argv[1]), output_dir=None, baseline_dir=None, '
                'embed_dir=run_demo.EMBED_DIR, cutoff="2021-06-01", horizon_days=30, max_users=100, '
                'fake_embeddings=fake); r = run_demo.build(a); '
                'print(hashlib.sha256(json.dumps(r["predictions"], sort_keys=True).encode()).hexdigest())')
        root = Path(__file__).resolve().parent.parent
        digests = set()
        for seed in ('1', '2'):
            done = subprocess.run([sys.executable, '-c', code, str(DATA)], cwd=root, check=True,
                                  capture_output=True, text=True,
                                  env={**__import__('os').environ, 'PYTHONHASHSEED': seed})
            digests.add(done.stdout.strip())
        self.assertEqual(len(digests), 1)

    def test_predictions_are_blind_to_rows_at_or_after_the_cutoff(self):
        """Delete every row dated at/after the cutoff: no list may change, truth must vanish."""
        with tempfile.TemporaryDirectory() as tmp:
            cut = Path(tmp)
            for name in ('transactions.csv', 'asset_information.csv',
                         'customer_information.csv', 'close_prices.csv'):
                with (DATA / name).open(encoding='utf-8-sig', newline='') as src, \
                        (cut / name).open('w', encoding='utf-8', newline='') as dst:
                    reader = csv.DictReader(src)
                    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    writer.writerows(r for r in reader if r['timestamp'][:10] < DEMO_CUTOFF)
            blind = run_demo.build(demo_args(cut, None, not HAVE_EMBEDDINGS))
        self.assertEqual(blind['users'], self.result['users'])
        self.assertEqual(blind['predictions'], self.result['predictions'])
        self.assertTrue(all(r['relevant_count'] == 0 for r in blind['per_user']))


@unittest.skipUnless(HAVE_DATA and HAVE_EMBEDDINGS, 'data or Kaggle embeddings missing')
class EmbeddingCoverage(unittest.TestCase):
    def test_every_eligible_demo_asset_has_both_embeddings(self):
        starter = run_demo.load_starter()
        assets = starter.load_data(DATA, DEMO_CUTOFF, 30)[0]
        tables, manifest = embeddings.load_tables()
        self.assertEqual(set(tables), {'llm_fill', 'e5_fill'})
        for table in tables.values():
            self.assertEqual(len(embeddings.vectors_for(assets, table)), len(assets))
        self.assertTrue(all(m['repeat_max_abs_diff'] == 0.0 and m['truncated_texts'] == 0
                            for m in manifest['models']))


if __name__ == '__main__':
    unittest.main()
