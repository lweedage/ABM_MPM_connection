"""Driver for the transmission ABM: runs X stochastic runs x X mobility seeds."""
from transmission_model.model import ModelT


def main():
    for run in range(10):
        for seed in range(1):
            params = {
                'savename': 'Medium100',
                'Ndays': 100,
                'initialization': 5,  # 4: beta = 0.5, 5: beta = 0.25 6: beta = 0.5, foi is exp.
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
