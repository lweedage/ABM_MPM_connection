"""Driver for the transmission ABM: runs X stochastic runs x X mobility seeds."""
from transmission_model.model import ModelT


def main():
    for run in range(5):
        for seed in range(10):
            params = {
                'savename': 'Medium',
                'Ndays': 100,
                'initialization': 4,
                'seed': seed,
            }
            model = ModelT(params)
            model.read_model_data()
            print('read empirical data')
            model.read_empirical_data()
            print('set parameters')
            model.set_parameters()
            print('initialise')
            model.initialise()
            print('simulate new')
            model.simulate_new()
            print('save run')
            model.save(run)


if __name__ == '__main__':
    main()
