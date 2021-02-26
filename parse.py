import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    head = ['network', 'batch', 'optimizer', 'mse', 'acc']
    csv = []
    csv.append(head)

    for line in lines:
        line.replace('\t', '')
        if line.startswith('network='):
            row = [
                'efnb4' if 'efnb4' in line else 'efnb3',
                '16' if '16' in line else '32' if '32' in line else '64',
                'rmsprop' if 'rmsprop' in line else 'Adam'
            ]