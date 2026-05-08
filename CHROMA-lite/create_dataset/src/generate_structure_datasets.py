import argparse

from random_layer import RandomLayerSimulation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--incidence_angle', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    num_layers = args.num_layers
    incidence_angle = args.incidence_angle
    seed = args.seed

    simulation = RandomLayerSimulation(num_layers=num_layers, incidence_angle=incidence_angle, seed=seed)
    simulation.run_simulation()
