class Metric:
    def __init__(self, metric_name):
        self.metric = load_metric(metric_name)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

        