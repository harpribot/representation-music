from experiment import Experiment


input_info = ('input', 5000)
output_info = [('output-1', 1), ('output-2', 1)]
exp = Experiment(input_info, output_info)
exp.initialize_network()

