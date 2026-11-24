import unittest
from vae.encoder_evaluation import controlled_data,alternatives,expression,OPERATIONS,equivalent_expression_metrics


class FrozenEvaluationProtocolTests(unittest.TestCase):
    def test_pairs_and_templates_are_held_out(self):
        rows=controlled_data()
        groups={split:{tuple(row['group']) for row in rows if row['split']==split}
                for split in ['train','validation','heldout']}
        self.assertFalse(groups['train']&groups['validation'])
        self.assertFalse(groups['train']&groups['heldout'])
        self.assertFalse(groups['validation']&groups['heldout'])
        self.assertTrue(all(row['template']==3 for row in rows if row['split']=='heldout'))
        self.assertTrue(all(row['template']!=3 for row in rows if row['split']!='heldout'))
        for split in groups:
            self.assertEqual({row['operation'] for row in rows if row['split']==split},set(OPERATIONS))
            self.assertEqual({row['a'] for row in rows if row['split']==split},set(range(1,21)))

    def test_commutative_swaps_are_not_counted_as_wrong_math(self):
        import torch
        labels=torch.tensor([[0,18,op] for op in range(4)])
        prediction=torch.tensor([[18,0,op] for op in range(4)])
        metrics=equivalent_expression_metrics(prediction,labels)
        self.assertEqual(metrics['mathematically_equivalent_expression_count'],2)

    def test_retrieval_has_no_identical_self_match(self):
        for row in controlled_data():
            if row['split']!='heldout':continue
            candidates=alternatives(row)
            self.assertEqual(len(set(candidates)),4)
            self.assertNotIn(row['text'],candidates)
            self.assertEqual(candidates[0],expression(row['a'],row['b'],row['operation'],0))


if __name__=='__main__':unittest.main()
