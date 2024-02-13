import re

# Define the file path
file_path = '/data/edacc_v1.0/dev/text_copy_edacc'

# Step 1: Insert comma after the last number in all rows of the text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify each line to insert a comma after the last number
modified_lines = []
for line in lines:
    numbers = re.findall(r'\d+', line)
    
    if numbers:
        last_number = numbers[-1]
        modified_line = line.replace(last_number, last_number + ',')
        modified_lines.append(modified_line)
    else:
        modified_lines.append(line)

# Step 2: Remove Second hyphen and replace with a comma
modified_lines_2 = []
for line in modified_lines:
    first_hyphen_index = line.find("-")
    
    if first_hyphen_index != -1:
        second_hyphen_index = line.find("-", first_hyphen_index + 1)
        
        if second_hyphen_index != -1:
            modified_line = line[:second_hyphen_index] + "," + line[second_hyphen_index + 1:]
            modified_lines_2.append(modified_line)
        else:
            modified_lines_2.append(line)
    else:
        modified_lines_2.append(line)

# Step 3: Remove zeroes
output_file_name = '/data/edacc_v1.0/dev/text_copy_edacc_check'

with open(output_file_name, 'w') as output_file:
    for line in modified_lines_2:
        pattern = r',0+'
        modified_line = re.sub(pattern, '-0', line)
        output_file.write(modified_line)

# Step 4: Add '.wav' extension at the end
with open(output_file_name, 'r') as file:
    lines = file.readlines()

# Modify each line to add ".wav" after the first comma
modified_lines_3 = []
for line in lines:
    first_comma_index = line.find(",")
    
    if first_comma_index != -1:
        modified_line = line[:first_comma_index] + ".wav" + line[first_comma_index:]
        modified_lines_3.append(modified_line)
    else:
        modified_lines_3.append(line)

# Write the modified content back to the file
with open(output_file_name, 'w') as file:
    file.writelines(modified_lines_3)

print("Combined modifications completed.")
