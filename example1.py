from vrptw_base import VrptwGraph
from multiple_ant_colony_system import MultipleAntColonySystem


if __name__ == '__main__':
    file_path = './solomon-100/c101.txt'
    ants_num = 10
    beta = 2
    q0 = 0.1
    show_figure = True
    max_iterations = 100  # Added max_iterations parameter

    graph = VrptwGraph(file_path)
    macs = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0, whether_or_not_to_show_figure=show_figure)
    macs.run_multiple_ant_colony_system(max_iterations=max_iterations)  # Pass max_iterations to run_multiple_ant_colony_system