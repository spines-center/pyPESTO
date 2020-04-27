import itertools
import tempfile
from typing import List, Tuple

# Zero-indexed column indices
MODEL_NAME_COLUMN = 0
SBML_FILENAME_COLUMN = 1
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0
PARAMETER_VALUE_DELIMITER = ';'

#def split_file_definition(line: str) -> Tuple:
#    '''
#    Returns a 3-tuple that contains the model name, SBML file name, and all
#    possible parameter values as a list (parameters) of tuples (parameter
#    values).
#    '''
#    columns = line.strip().split('\t')
#    return (columns[MODEL_NAME_COLUMN],
#            columns[SBML_FILENAME_COLUMN],
#            [definition.split(PARAMETER_VALUE_DELIMITER)
#                for definition in columns[PARAMETER_DEFINITIONS_START:]])
#
#def generate_model_definition_lines(
#        model_name: str,
#        sbml_filename: str,
#        parameter_definitions: List[str]
#) -> str:
#    '''Yields all expanded model definitions.'''
#    for index, selection in enumerate(
#            itertools.product(*parameter_definitions)):
#        yield model_name+f'_{index}' + '\t' + sbml_filename + '\t' + \
#                '\t'.join(selection) + '\n'
#
#def unpack_file(file_name: str):
#    '''
#    Converts model definitions from the compressed form to the expanded form.
#    '''
#    expanded_models_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
#    with open(file_name) as fh:
#        for line_index, line in enumerate(fh):
#            if line_index != HEADER_ROW:
#                for definition_line in generate_model_definition_lines(
#                        *split_file_definition(line)):
#                    expanded_models_file.write(definition_line)
#            else:
#                expanded_models_file.write(line)
#    return expanded_models_file

def unpack_file(file_name: str):
    '''
    Unpacks a model definition file into a new temporary file that is returned.
    '''
    expanded_models_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(file_name) as fh:
        for line_index, line in enumerate(fh):
            if line_index != HEADER_ROW:
                columns = line.strip().split('\t')
                parameter_definitions = [
                    definition.split(PARAMETER_VALUE_DELIMITER)
                    for definition in columns[PARAMETER_DEFINITIONS_START:]
                ]
                for index, selection in enumerate(itertools.product(
                        *parameter_definitions
                )):
                    expanded_models_file.write(
                        '\t'.join([
                            columns[MODEL_NAME_COLUMN]+f'_{index}',
                            columns[SBML_FILENAME_COLUMN],
                            *selection
                        ]) + '\n'
                    )
            else:
                expanded_models_file.write(line)
    return expanded_models_file




original_file = 'example_model_selection_definitions.tsv'
unpacked_file = unpack_file(original_file)

# The file at expanded_models_file.name now contains the expanded model.
print('The expanded model space definitions have been stored in: ' + \
        unpacked_file.name)
