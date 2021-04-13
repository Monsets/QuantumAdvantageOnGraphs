config = {

    # ============ GENERAL SETTINGS ============ #

    'random_seed': 0,

    # Generate graphs or not

    'include_random_graphs': True,
    'include_path_graphs': False,

    # ============ PATH GRAPH ============ #


    # Path graphs will be generated with number of nodes in n = 3, ..., max_node_path_graph
    'max_node_for_path_graph': 7,

    # ============ RANDOM GRAPH ============ #

    # Path graphs will be generated with number of nodes in n = 3, ..., max_node_path_graph
    'random_graphs_amount': 2000,
    'max_node_for_random_graph': 16,
    'min_node_for_random_graph': 15,


    # ============ DATA PATHS ============ #

    'data_path': 'data',
    'adjacency_path': 'adjacency.pkl',
    'rank_path': 'y.pkl',
    'path_graph_path': 'path_graphs',
    'path_random_path': 'random_graphs',
    'node_feature_path': 'node_features.pkl',

    # ============ RANDOM WALKS SETTINGS ============ #

    # Steps per seconds. For continious random walk time is t = dt * step_number and step_number is number_of_step // dt
    'dt': 0.01,
    # Number of steps for which walks will be running. If None of the walk has succeeded then the graph is not added
    'number_of_steps': 100,

    # Initial and target nodes which are used for the walks
    'initial_node': 0,
    'target_node': 1,

    # ============ CQCNN CONFIG ============ #

    # CQNN model config
    'cqnn_epochs': 3000,
    'cqnn_samples_per_epoch': 100,
}