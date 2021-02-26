import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()

    head = ['network', 'batch', 'optimizer', 'mse', 'acc', 'seconds']
    csv = []
    csv.append(head)

    row = None
    time = 0

    for line in lines:

        if '=] - ' in line:
            time += int(line.split('s')[0].split(' ')[-1])
            #print('hi', time)

        if 'network=' in line:
            row = [
                'efnb4' if 'efnb4' in line else 'efnb3',
                '16' if '16' in line else '32' if '32' in line else '64',
                'RMSprop' if 'RMSprop' in line else 'Adam'
            ]

        if 'accuracy=' in line:
            items = [item.split('=')[1] for item in line.split(', ')]
            row.append(items[0])
            row.append(items[1].replace('\n', ''))
            row.append(str(time))
            csv.append(row)
            time = 0


    print(csv)

    with open('eye.csv', 'w') as file:
        file.write('\n'.join([','.join(row) for row in csv]))