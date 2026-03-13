from hgere.ner_classifier.train import run_train_span_classifier, MODEL_CLASSES  # noqa: F401


def main(args=None):
    run_train_span_classifier(args)


if __name__ == "__main__":
    main()
