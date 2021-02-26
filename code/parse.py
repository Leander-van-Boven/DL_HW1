import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    head = ['network', 'batch', 'optimizer', 'mse', 'acc', 'seconds',
            'epochs', 'tpe']
    csv = []
    csv.append(head)

    row = None
    time = 0
    epochs = 0

    for line in lines:

        if '=] - ' in line:
            time += int(line.split('s')[0].split(' ')[-1])
            epochs += 1

        if 'network=' in line:
            row = [
                'efnb4' if 'efnb4' in line else 'efnb3',
                '16' if '16' in line else '32' if '32' in line else '64',
                'RMSprop' if 'RMSprop' in line else 'Adam'
            ]
            epochs = 0

        if 'accuracy=' in line:
            items = [item.split('=')[1] for item in line.split(', ')]
            row.append(items[0])
            row.append(items[1].replace('\n', ''))
            row.append(str(time))
            row.append(str(epochs))
            row.append(str(time/epochs))
            csv.append(row)
            time = 0
            epochs = 0

    print(csv)

    with open('eye.csv', 'w') as file:
        file.write('\n'.join([','.join(row) for row in csv]))
