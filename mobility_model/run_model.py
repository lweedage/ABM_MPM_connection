"""Driver for the mobility model: positions people over 10 mobility seeds."""
from mobility_model.model import ModelM


def main():
    params = {
        'savename': 'High30',
        'division': 100,  # 5000 / 1000 / 500 / 100
        'use_same_mobility': True,
    }

    model = ModelM(params)
    print('Read data')
    model.read_data()
    print('Position people...')
    for mc in range(10):
        model.position_people(mc)
        print('Save')
        model.save(mc)


if __name__ == '__main__':
    main()
