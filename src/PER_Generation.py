def calculate_phoneme_errors(expected, predicted):
    len_ref = len(expected)
    len_hyp = len(predicted)

    # Initialize counters for insertion, deletion, and substitution
    insertions, deletions, substitutions = 0, 0, 0

    # Initialize the confusion matrix
    confusion_matrix = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]

    # Fill the confusion matrix using dynamic programming
    for i in range(len_ref + 1):
        for j in range(len_hyp + 1):
            if i == 0:
                confusion_matrix[i][j] = j
            elif j == 0:
                confusion_matrix[i][j] = i
            elif (
                expected[i - 1] == predicted[j - 1]
                or expected[i - 1] == '-'
                or predicted[j - 1] == '-'
            ):
                confusion_matrix[i][j] = confusion_matrix[i - 1][j - 1]
            else:
                confusion_matrix[i][j] = 1 + min(
                    confusion_matrix[i - 1][j],
                    confusion_matrix[i][j - 1],
                    confusion_matrix[i - 1][j - 1],
                )

    # Traceback to count insertions, deletions, and substitutions
    i, j = len_ref, len_hyp
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i - 1] == predicted[j - 1]:
            i -= 1
            j -= 1
        elif j > 0 and (
            i == 0
            or confusion_matrix[i][j - 1]
            <= min(confusion_matrix[i - 1][j], confusion_matrix[i - 1][j - 1])
        ):
            insertions += 1
            j -= 1
        elif i > 0 and (
            j == 0
            or confusion_matrix[i - 1][j]
            <= min(confusion_matrix[i][j - 1], confusion_matrix[i - 1][j - 1])
        ):
            deletions += 1
            i -= 1
        else:
            substitutions += 1
            i -= 1
            j -= 1

    # Total number of phonemes in the reference sequence
    total_phonemes = len_ref
    per = (insertions + deletions + substitutions) / total_phonemes

    return insertions, deletions, substitutions, total_phonemes, per


def calculate_phoneme_error_rate(reference, predicted):
    len_ref = len(reference)
    len_hyp = len(predicted)

    # Initialize the confusion matrix
    confusion_matrix = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]

    # Fill the confusion matrix using dynamic programming
    for i in range(len_ref + 1):
        for j in range(len_hyp + 1):
            if i == 0:
                confusion_matrix[i][j] = j
            elif j == 0:
                confusion_matrix[i][j] = i
            elif j > 0 and i > 0 and reference[i - 1] == predicted[j - 1]:
                confusion_matrix[i][j] = confusion_matrix[i - 1][j - 1]
            else:
                confusion_matrix[i][j] = 1 + min(
                    confusion_matrix[i - 1][j],
                    confusion_matrix[i][j - 1],
                    confusion_matrix[i - 1][j - 1],
                )

    # Phoneme Error Rate (PER) calculation
    phoneme_error_rate = confusion_matrix[len_ref][len_hyp] / len_ref
    return phoneme_error_rate


# Example Usage:
