from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


class RagasEvaluation:
    def __init__(self, results, llm):
        self.results = results
        self.llm = llm

    def evaluate(self):
        dataset = Dataset.from_list(self.results)
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            llm=self.llm
        )

        return result


def evaluate_ragas(dataset):

    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )

    return result
