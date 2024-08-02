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
        dataset = Dataset.from_dict(self.results)
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

dataset = load_dataset("explodinggradients/amnesty_qa", "english_v2")
print(evaluate_ragas(dataset["eval"]))