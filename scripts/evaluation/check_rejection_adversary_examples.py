#!/usr/bin/env python

import argparse
import logging

from flowspeech.DL_speech_tech import default_model
from flowspeech.performance_functions import check_acceptance_adversaries

logging.basicConfig()
logger = logging.getLogger("check_rejection_adversary_examples")
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Validate the behaviour of the rejection method"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./plots/adversary_rejection_results.csv",
        help="Output file for results",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    model = default_model

    data = check_acceptance_adversaries(model)

    logger.info(f"Saving results to {args.output}")
    data.to_csv(args.output)

    data = data[
        [
            "exp_phonetics",
            "detected_phones",
            "audio_status_message",
            "speech_rate",
            "per_aligned",
            "audio_file_url",
            "rejected",
            "should_reject",
        ]
    ]

    false_negatives = data[data.rejected & ~data.should_reject]
    logger.info(
        f"False negatives: {len(false_negatives)} / {len(data)} = {100 * len(false_negatives) / len(data):.2f}%"
    )
    print(false_negatives)  # noqa: T201

    false_positives = data[~data.rejected & data.should_reject]
    logger.info(
        f"False positives: {len(false_positives)} / {len(data)} = {100 * len(false_positives) / len(data):.2f}%"
    )
    print(false_positives)  # noqa: T201


if __name__ == "__main__":
    main()
