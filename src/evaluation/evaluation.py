from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


class RagasEvaluation:

    def __init__(self, responses):
        self.responses = responses # [{"query": , "response": , "sourced_documents": }]
        self.eval_chains = {
            m.name: RagasEvaluatorChain(metric=m)
            for m in [faithfulness, answer_relevancy, context_precision, context_recall]
        }

    def evaluate(self):
        """Returns the aggregated evaluation results for all responses."""
        results = {m.name: [] for m in self.eval_chains.values()}
        for response in self.responses:
            for name, eval_chain in self.eval_chains.items():
                results[name].append(eval_chain.evaluate(response))
        # Aggregate the results
        return {
            name: sum(scores) / len(scores)
            for name, scores in results.items()
        }
