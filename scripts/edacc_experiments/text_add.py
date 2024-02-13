def add_period_to_lines(text):
    # Add a period to the end of each line
    lines_with_period = [line.strip() + '.' for line in text.split('\n')]

    # Join the lines back into a single string
    modified_text = '\n'.join(lines_with_period)

    return modified_text


def save_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)


# Read the input text file
input_file = '/home/mambauser/code/data/edacc_v1.0/test/text_edited_new'
with open(input_file, 'r') as file:
    input_text = file.read()

# Add a period to each line
modified_text = add_period_to_lines(input_text)

# Specify the output file name
output_file = '/home/mambauser/code/data/edacc_v1.0/test/text_edited_new_2'

# Save the modified text to a file
save_to_file(modified_text, output_file)

print(f"Modified text saved to '{output_file}'.")
